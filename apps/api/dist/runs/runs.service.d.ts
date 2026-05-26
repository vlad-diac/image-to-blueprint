import { RunpodService } from '../runpod/runpod.service';
import { PrismaService } from '../prisma/prisma.service';
import { type CreateRunFields, type RunSerializable } from './runs.helpers';
export declare class RunsService {
    private readonly prisma;
    private readonly runpod;
    private readonly logger;
    constructor(prisma: PrismaService, runpod: RunpodService);
    listRecent(limit?: number): Promise<RunSerializable[]>;
    findOne(id: string, refresh?: boolean): Promise<RunSerializable>;
    getInputBytes(id: string): Promise<Buffer>;
    getOutputBytes(id: string): Promise<Buffer | null>;
    createWithImage(buffer: Buffer | undefined, fields: CreateRunFields): Promise<RunSerializable>;
    cancel(id: string): Promise<RunSerializable>;
    reconcile(runId: string, jobId: string): Promise<void>;
    private applyStatusPayload;
    private maybeSetStartedAt;
    sweepStaleInFlight(limit?: number): Promise<void>;
}
