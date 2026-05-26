"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var RunsService_1;
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunsService = void 0;
const common_1 = require("@nestjs/common");
const client_1 = require("@prisma/client");
const runpod_service_1 = require("../runpod/runpod.service");
const prisma_service_1 = require("../prisma/prisma.service");
const runs_helpers_1 = require("./runs.helpers");
function toSerializablePartial(r) {
    const full = r;
    return (0, runs_helpers_1.serializeRun)(full);
}
let RunsService = RunsService_1 = class RunsService {
    constructor(prisma, runpod) {
        this.prisma = prisma;
        this.runpod = runpod;
        this.logger = new common_1.Logger(RunsService_1.name);
    }
    async listRecent(limit = 20) {
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
    async findOne(id, refresh = true) {
        let run = await this.prisma.run.findUnique({
            where: { id },
        });
        if (!run)
            throw new common_1.NotFoundException(`Run ${id} not found`);
        if (refresh && !(0, runs_helpers_1.isTerminalRunStatus)(run.status) && run.runpodJobId) {
            await this.reconcile(run.id, run.runpodJobId);
            run = await this.prisma.run.findUniqueOrThrow({ where: { id } });
        }
        return (0, runs_helpers_1.serializeRun)(run);
    }
    async getInputBytes(id) {
        const row = await this.prisma.run.findUnique({
            where: { id },
            select: { inputImage: true },
        });
        if (!row)
            throw new common_1.NotFoundException();
        return Buffer.from(row.inputImage);
    }
    async getOutputBytes(id) {
        const row = await this.prisma.run.findUnique({
            where: { id },
            select: { status: true, outputImage: true },
        });
        if (!row)
            throw new common_1.NotFoundException();
        if (!row.outputImage || row.status !== client_1.RunStatus.SUCCEEDED)
            return null;
        return Buffer.from(row.outputImage);
    }
    async createWithImage(buffer, fields) {
        if (!buffer?.length) {
            throw new common_1.BadRequestException('image file is required');
        }
        const positive = (fields.positivePrompt ?? '').trim();
        if (!positive) {
            throw new common_1.BadRequestException('positivePrompt is required');
        }
        const steps = fields.steps ?? 4;
        const cfg = fields.cfg ?? 1.0;
        if (steps < 1 || steps > 100) {
            throw new common_1.BadRequestException('steps must be between 1 and 100');
        }
        const seed = fields.seed !== undefined && fields.seed !== null
            ? BigInt(Math.trunc(Number(fields.seed)))
            : undefined;
        const run = await this.prisma.run.create({
            data: {
                status: client_1.RunStatus.QUEUED,
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
                seed: fields.seed !== undefined && fields.seed !== null
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
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            this.logger.warn(`RunPod submit failed run=${run.id}: ${msg}`);
            await this.prisma.run.update({
                where: { id: run.id },
                data: {
                    status: client_1.RunStatus.FAILED,
                    errorMessage: msg,
                    completedAt: new Date(),
                },
            });
        }
        const finalRun = await this.prisma.run.findUniqueOrThrow({ where: { id: run.id } });
        return (0, runs_helpers_1.serializeRun)(finalRun);
    }
    async cancel(id) {
        const run = await this.prisma.run.findUnique({ where: { id } });
        if (!run)
            throw new common_1.NotFoundException();
        if (!run.runpodJobId)
            throw new common_1.BadRequestException('No RunPod job id');
        if ((0, runs_helpers_1.isTerminalRunStatus)(run.status)) {
            return (0, runs_helpers_1.serializeRun)(run);
        }
        try {
            await this.runpod.cancel(run.runpodJobId);
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            this.logger.warn(`RunPod cancel failed run=${id}: ${msg}`);
        }
        if (run.runpodJobId) {
            await this.reconcile(run.id, run.runpodJobId);
        }
        const updated = await this.prisma.run.findUniqueOrThrow({ where: { id } });
        return (0, runs_helpers_1.serializeRun)(updated);
    }
    async reconcile(runId, jobId) {
        const active = await this.prisma.run.findFirst({
            where: {
                id: runId,
                status: {
                    in: [client_1.RunStatus.QUEUED, client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS],
                },
            },
        });
        if (!active)
            return;
        let st;
        try {
            st = await this.runpod.status(jobId);
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            this.logger.warn(`RunPod status failed run=${runId}: ${msg}`);
            return;
        }
        await this.applyStatusPayload(active.id, st);
    }
    async applyStatusPayload(runId, st) {
        const raw = typeof st === 'object' && st !== null
            ? JSON.parse(JSON.stringify(st))
            : client_1.Prisma.JsonNull;
        const delayMs = typeof st.delayTime === 'number' ? Math.round(st.delayTime) : undefined;
        const executionMs = typeof st.executionTime === 'number' ? Math.round(st.executionTime) : undefined;
        if (st.status === 'IN_QUEUE') {
            await this.prisma.run.updateMany({
                where: {
                    id: runId,
                    status: {
                        in: [client_1.RunStatus.QUEUED, client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS],
                    },
                },
                data: {
                    status: client_1.RunStatus.IN_QUEUE,
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
                        in: [client_1.RunStatus.QUEUED, client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS],
                    },
                },
                data: {
                    status: client_1.RunStatus.IN_PROGRESS,
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
            let outputBuf;
            if (parsed?.image_b64) {
                try {
                    outputBuf = Buffer.from(parsed.image_b64, 'base64');
                }
                catch {
                    outputBuf = undefined;
                }
            }
            const duration = delayMs !== undefined || executionMs !== undefined
                ? (delayMs ?? 0) + (executionMs ?? 0)
                : undefined;
            await this.prisma.run.updateMany({
                where: {
                    id: runId,
                    status: {
                        in: [client_1.RunStatus.QUEUED, client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS],
                    },
                },
                data: {
                    status: client_1.RunStatus.SUCCEEDED,
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
        if (st.status === 'FAILED' ||
            st.status === 'CANCELLED' ||
            st.status === 'TIMED_OUT') {
            const mapped = st.status === 'FAILED'
                ? client_1.RunStatus.FAILED
                : st.status === 'CANCELLED'
                    ? client_1.RunStatus.CANCELLED
                    : client_1.RunStatus.TIMED_OUT;
            await this.prisma.run.updateMany({
                where: {
                    id: runId,
                    status: {
                        in: [client_1.RunStatus.QUEUED, client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS],
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
    async maybeSetStartedAt(runId) {
        await this.prisma.$executeRaw `
      UPDATE "Run"
      SET "startedAt" = COALESCE("startedAt", NOW())
      WHERE id = ${runId} AND "startedAt" IS NULL
    `;
    }
    async sweepStaleInFlight(limit = 8) {
        const cutoff = new Date(Date.now() - 1500);
        const stale = await this.prisma.run.findMany({
            where: {
                status: { in: [client_1.RunStatus.IN_QUEUE, client_1.RunStatus.IN_PROGRESS] },
                runpodJobId: { not: null },
                updatedAt: { lt: cutoff },
            },
            take: limit,
            select: { id: true, runpodJobId: true },
        });
        await Promise.all(stale.map((r) => r.runpodJobId ? this.reconcile(r.id, r.runpodJobId) : Promise.resolve()));
    }
};
exports.RunsService = RunsService;
exports.RunsService = RunsService = RunsService_1 = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService,
        runpod_service_1.RunpodService])
], RunsService);
function mapIncomingStatus(s) {
    if (s === 'IN_PROGRESS')
        return client_1.RunStatus.IN_PROGRESS;
    return client_1.RunStatus.IN_QUEUE;
}
function coerceHandlerOutput(raw) {
    if (!raw)
        return null;
    if (typeof raw === 'string') {
        try {
            return coerceHandlerOutput(JSON.parse(raw));
        }
        catch {
            return null;
        }
    }
    const o = raw;
    if (o.output && typeof o.output === 'object') {
        return coerceHandlerOutput(o.output);
    }
    return raw;
}
//# sourceMappingURL=runs.service.js.map