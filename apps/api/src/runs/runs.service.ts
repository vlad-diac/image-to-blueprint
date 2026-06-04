import {
  BadRequestException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { Prisma, RunStatus } from '@prisma/client';
import { RunpodService } from '../runpod/runpod.service';
import type { RunpodHandlerOutput, RunpodJobStatus } from '../runpod/runpod.types';
import { PrismaService } from '../prisma/prisma.service';
import {
  type CreateRunFields,
  isTerminalRunStatus,
  serializeRun,
  type RunSerializable,
} from './runs.helpers';

const OUTPUT_META_SELECT = {
  select: {
    index: true,
    seed: true,
    width: true,
    height: true,
  },
  orderBy: { index: 'asc' as const },
};

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
        numImages: true,
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
        outputs: OUTPUT_META_SELECT,
      },
    });
    return rows.map((r) => serializeRun(r));
  }

  async findOne(id: string, refresh = true): Promise<RunSerializable> {
    let run = await this.prisma.run.findUnique({
      where: { id },
      include: { outputs: OUTPUT_META_SELECT },
    });
    if (!run) throw new NotFoundException(`Run ${id} not found`);

    if (refresh && !isTerminalRunStatus(run.status) && run.runpodJobId) {
      await this.reconcile(run.id, run.runpodJobId);
      run = await this.prisma.run.findUniqueOrThrow({
        where: { id },
        include: { outputs: OUTPUT_META_SELECT },
      });
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

  async getOutputBytesAt(runId: string, index: number): Promise<Buffer | null> {
    const row = await this.prisma.runImage.findUnique({
      where: { runId_index: { runId, index } },
      include: { run: { select: { status: true } } },
    });
    if (!row?.run || row.run.status !== RunStatus.SUCCEEDED) return null;
    return Buffer.from(row.bytes);
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
    const numImages = Math.min(
      Math.max(fields.numImages ?? 3, 1),
      4,
    );
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
        numImages,
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
        num_images: numImages,
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

    const finalRun = await this.prisma.run.findUniqueOrThrow({
      where: { id: run.id },
      include: { outputs: OUTPUT_META_SELECT },
    });
    return serializeRun(finalRun);
  }

  async cancel(id: string): Promise<RunSerializable> {
    const run = await this.prisma.run.findUnique({
      where: { id },
      include: { outputs: OUTPUT_META_SELECT },
    });
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
    const updated = await this.prisma.run.findUniqueOrThrow({
      where: { id },
      include: { outputs: OUTPUT_META_SELECT },
    });
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
      const duration =
        delayMs !== undefined || executionMs !== undefined
          ? (delayMs ?? 0) + (executionMs ?? 0)
          : undefined;

      const rows = extractOutputImageRows(parsed);
      if (!rows.length) {
        await this.prisma.run.updateMany({
          where: {
            id: runId,
            status: {
              in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
            },
          },
          data: {
            status: RunStatus.FAILED,
            errorMessage: 'RunPod completed but no decodeable images in output',
            rawStatus: raw,
            completedAt: new Date(),
          },
        });
        await this.maybeSetStartedAt(runId);
        return;
      }

      await this.prisma.$transaction(async (tx) => {
        const alive = await tx.run.findFirst({
          where: {
            id: runId,
            status: {
              in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
            },
          },
        });
        if (!alive) return;

        await tx.runImage.deleteMany({ where: { runId } });

        const sorted = [...rows].sort((a, b) => a.index - b.index);
        for (const r of sorted) {
          await tx.runImage.create({
            data: {
              runId,
              index: r.index,
              bytes: r.buf,
              seed: r.seed ?? null,
              width: r.width,
              height: r.height,
            },
          });
        }

        const numStored = sorted.length;

        await tx.run.updateMany({
          where: {
            id: runId,
            status: {
              in: [RunStatus.QUEUED, RunStatus.IN_QUEUE, RunStatus.IN_PROGRESS],
            },
          },
          data: {
            status: RunStatus.SUCCEEDED,
            numImages: numStored,
            workerJobDir: parsed?.job_dir ?? null,
            durationMs: duration ?? null,
            delayMs,
            executionMs,
            rawStatus: raw,
            completedAt: new Date(),
            errorMessage: null,
          },
        });
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

interface DecodedImgRow {
  index: number;
  buf: Buffer;
  seed: bigint | undefined;
  width: number | null;
  height: number | null;
}

/** Decode RunPod worker output → DB rows for RunImage (no inserts here). */
function extractOutputImageRows(parsed: RunpodHandlerOutput | null): DecodedImgRow[] {
  if (!parsed) return [];
  const w = parsed.width ?? null;
  const h = parsed.height ?? null;
  const list = parsed.images;
  if (list?.length) {
    const out: DecodedImgRow[] = [];
    for (const im of [...list].sort((a, b) => (a.index ?? 0) - (b.index ?? 0))) {
      if (
        typeof im.image_b64 !== 'string' ||
        typeof im.index !== 'number' ||
        im.index < 0
      ) {
        continue;
      }
      let buf: Buffer;
      try {
        buf = Buffer.from(im.image_b64, 'base64');
      } catch {
        continue;
      }
      if (!buf.length) continue;
      let seed: bigint | undefined;
      if (im.seed !== undefined && im.seed !== null) {
        try {
          seed = BigInt(Math.trunc(Number(im.seed)));
        } catch {
          seed = undefined;
        }
      }
      out.push({
        index: im.index,
        buf,
        seed,
        width: w,
        height: h,
      });
    }
    return out;
  }
  if (parsed.image_b64) {
    try {
      const buf = Buffer.from(parsed.image_b64, 'base64');
      if (!buf.length) return [];
      return [
        {
          index: 0,
          buf,
          seed: undefined,
          width: w,
          height: h,
        },
      ];
    } catch {
      return [];
    }
  }
  return [];
}
