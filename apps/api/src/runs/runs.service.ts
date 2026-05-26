import {
  BadRequestException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { Prisma, RunStatus, type Run } from '@prisma/client';
import { RunpodService } from '../runpod/runpod.service';
import type { RunpodHandlerOutput, RunpodJobStatus } from '../runpod/runpod.types';
import { PrismaService } from '../prisma/prisma.service';
import {
  type CreateRunFields,
  isTerminalRunStatus,
  serializeRun,
  type RunSerializable,
} from './runs.helpers';

type RunListRow = Omit<Run, 'inputImage' | 'outputImage'>;

function toSerializablePartial(r: RunListRow): RunSerializable {
  const full = r as unknown as Run;
  return serializeRun(full);
}

@Injectable()
export class RunsService {
  private readonly logger = new Logger(RunsService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly runpod: RunpodService,
  ) {}

  async listRecent(limit = 20): Promise<RunSerializable[]> {
    const rows = await this.prisma.run.findMany({
      take: Math.min(Math.max(limit, 1), 100),
      orderBy: { updatedAt: 'desc' },
      select: {
        id: true,
        status: true,
        positivePrompt: true,
        negativePrompt: true,
        steps: true,
        cfg: true,
        seed: true,
        runpodJobId: true,
        workerJobDir: true,
        durationMs: true,
        delayMs: true,
        executionMs: true,
        errorMessage: true,
        rawStatus: true,
        startedAt: true,
        completedAt: true,
        createdAt: true,
        updatedAt: true,
      },
    });
    return rows.map(toSerializablePartial);
  }

  async findOne(id: string, refresh = true): Promise<RunSerializable> {
    let run = await this.prisma.run.findUnique({
      where: { id },
    });
    if (!run) throw new NotFoundException(`Run ${id} not found`);

    if (refresh && !isTerminalRunStatus(run.status) && run.runpodJobId) {
      await this.reconcile(run.id, run.runpodJobId);
      run = await this.prisma.run.findUniqueOrThrow({ where: { id } });
    }
    return serializeRun(run);
  }

  async getInputBytes(id: string): Promise<Buffer> {
    const row = await this.prisma.run.findUnique({
      where: { id },
      select: { inputImage: true },
    });
    if (!row) throw new NotFoundException();
    return Buffer.from(row.inputImage);
  }

  async getOutputBytes(id: string): Promise<Buffer | null> {
    const row = await this.prisma.run.findUnique({
      where: { id },
      select: { status: true, outputImage: true },
    });
    if (!row) throw new NotFoundException();
    if (!row.outputImage || row.status !== RunStatus.SUCCEEDED) return null;
    return Buffer.from(row.outputImage);
  }

  async createWithImage(
    buffer: Buffer | undefined,
    fields: CreateRunFields,
  ): Promise<RunSerializable> {
    if (!buffer?.length) {
      throw new BadRequestException('image file is required');
    }
    const positive = (fields.positivePrompt ?? '').trim();
    if (!positive) {
      throw new BadRequestException('positivePrompt is required');
    }
    const steps = fields.steps ?? 4;
    const cfg = fields.cfg ?? 1.0;
    if (steps < 1 || steps > 100) {
      throw new BadRequestException('steps must be between 1 and 100');
    }
    const seed =
      fields.seed !== undefined && fields.seed !== null
        ? BigInt(Math.trunc(Number(fields.seed)))
        : undefined;

    const run = await this.prisma.run.create({
      data: {
        status: RunStatus.QUEUED,
        positivePrompt: positive,
        negativePrompt: (fields.negativePrompt ?? '').trim(),
        steps,
        cfg,
        seed,
        inputImage: buffer,
      },
    });

    const image_b64 = buffer.toString('base64');
    try {
      const sub = await this.runpod.submit({
        image_b64,
        positive_prompt: positive,
        negative_prompt: fields.negativePrompt ?? '',
        steps,
        cfg,
        seed:
          fields.seed !== undefined && fields.seed !== null
            ? Math.trunc(Number(fields.seed))
            : undefined,
      });
      const rpStatus = mapIncomingStatus(sub.status ?? 'IN_QUEUE');
      await this.prisma.run.update({
        where: { id: run.id },
        data: {
          runpodJobId: sub.id,
          status: rpStatus,
        },
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.logger.warn(`RunPod submit failed run=${run.id}: ${msg}`);
      await this.prisma.run.update({
        where: { id: run.id },
        data: {
          status: RunStatus.FAILED,
          errorMessage: msg,
          completedAt: new Date(),
        },
      });
    }

    const finalRun = await this.prisma.run.findUniqueOrThrow({ where: { id: run.id } });
    return serializeRun(finalRun);
  }

  async cancel(id: string): Promise<RunSerializable> {
    const run = await this.prisma.run.findUnique({ where: { id } });
    if (!run) throw new NotFoundException();
    if (!run.runpodJobId) throw new BadRequestException('No RunPod job id');
    if (isTerminalRunStatus(run.status)) {
      return serializeRun(run);
    }
    try {
      await this.runpod.cancel(run.runpodJobId);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.logger.warn(`RunPod cancel failed run=${id}: ${msg}`);
    }
    if (run.runpodJobId) {
      await this.reconcile(run.id, run.runpodJobId);
    }
    const updated = await this.prisma.run.findUniqueOrThrow({ where: { id } });
    return serializeRun(updated);
  }

  async reconcile(runId: string, jobId: string): Promise<void> {
    const active = await this.prisma.run.findFirst({
      where: {
        id: runId,
        status: {
          in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
        },
      },
    });
    if (!active) return;

    let st: Awaited<ReturnType<RunpodService['status']>>;
    try {
      st = await this.runpod.status(jobId);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.logger.warn(`RunPod status failed run=${runId}: ${msg}`);
      return;
    }

    await this.applyStatusPayload(active.id, st);
  }

  private async applyStatusPayload(
    runId: string,
    st: {
      status: RunpodJobStatus;
      delayTime?: number;
      executionTime?: number;
      output?: unknown;
      error?: string;
    },
  ): Promise<void> {
    const raw =
      typeof st === 'object' && st !== null
        ? (JSON.parse(JSON.stringify(st)) as Prisma.InputJsonValue)
        : Prisma.JsonNull;
    const delayMs =
      typeof st.delayTime === 'number' ? Math.round(st.delayTime) : undefined;
    const executionMs =
      typeof st.executionTime === 'number' ? Math.round(st.executionTime) : undefined;

    if (st.status === 'IN_QUEUE') {
      await this.prisma.run.updateMany({
        where: {
          id: runId,
          status: {
            in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
          },
        },
        data: {
          status: RunStatus.IN_QUEUE,
          delayMs,
          executionMs,
          rawStatus: raw,
        },
      });
      return;
    }

    if (st.status === 'IN_PROGRESS') {
      await this.prisma.run.updateMany({
        where: {
          id: runId,
          status: {
            in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
          },
        },
        data: {
          status: RunStatus.IN_PROGRESS,
          delayMs,
          executionMs,
          rawStatus: raw,
        },
      });
      await this.maybeSetStartedAt(runId);
      return;
    }

    if (st.status === 'COMPLETED') {
      const parsed = coerceHandlerOutput(st.output);
      let outputBuf: Buffer | undefined;
      if (parsed?.image_b64) {
        try {
          outputBuf = Buffer.from(parsed.image_b64, 'base64');
        } catch {
          outputBuf = undefined;
        }
      }
      const duration =
        delayMs !== undefined || executionMs !== undefined
          ? (delayMs ?? 0) + (executionMs ?? 0)
          : undefined;

      await this.prisma.run.updateMany({
        where: {
          id: runId,
          status: {
            in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
          },
        },
        data: {
          status: RunStatus.SUCCEEDED,
          outputImage: outputBuf,
          workerJobDir: parsed?.job_dir ?? null,
          durationMs: duration ?? null,
          delayMs,
          executionMs,
          rawStatus: raw,
          completedAt: new Date(),
          errorMessage: null,
        },
      });
      await this.maybeSetStartedAt(runId);
      return;
    }

    if (
      st.status === 'FAILED' ||
      st.status === 'CANCELLED' ||
      st.status === 'TIMED_OUT'
    ) {
      const mapped =
        st.status === 'FAILED'
          ? RunStatus.FAILED
          : st.status === 'CANCELLED'
            ? RunStatus.CANCELLED
            : RunStatus.TIMED_OUT;
      await this.prisma.run.updateMany({
        where: {
          id: runId,
          status: {
            in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
          },
        },
        data: {
          status: mapped,
          errorMessage: st.error ?? String(mapped),
          rawStatus: raw,
          completedAt: new Date(),
        },
      });
      return;
    }
  }

  private async maybeSetStartedAt(runId: string): Promise<void> {
    await this.prisma.$executeRaw`
      UPDATE "Run"
      SET "startedAt" = COALESCE("startedAt", NOW())
      WHERE id = ${runId} AND "startedAt" IS NULL
    `;
  }

  async sweepStaleInFlight(limit = 8): Promise<void> {
    const cutoff = new Date(Date.now() - 1500);
    const stale = await this.prisma.run.findMany({
      where: {
        status: { in: [RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS] },
        runpodJobId: { not: null },
        updatedAt: { lt: cutoff },
      },
      take: limit,
      select: { id: true, runpodJobId: true },
    });
    await Promise.all(
      stale.map((r) =>
        r.runpodJobId ? this.reconcile(r.id, r.runpodJobId) : Promise.resolve(),
      ),
    );
  }
}

function mapIncomingStatus(s: RunpodJobStatus | string): RunStatus {
  if (s === 'IN_PROGRESS') return RunStatus.IN_PROGRESS;
  return RunStatus.IN_QUEUE;
}

function coerceHandlerOutput(raw: unknown): RunpodHandlerOutput | null {
  if (!raw) return null;
  if (typeof raw === 'string') {
    try {
      return coerceHandlerOutput(JSON.parse(raw) as unknown);
    } catch {
      return null;
    }
  }
  const o = raw as Record<string, unknown>;
  if (o.output && typeof o.output === 'object') {
    return coerceHandlerOutput(o.output);
  }
  return raw as RunpodHandlerOutput;
}
