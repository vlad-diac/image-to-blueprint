import { RunStatus, type Run } from '@prisma/client';
export declare function isTerminalRunStatus(s: RunStatus): boolean;
export interface RunSerializable {
    id: string;
    status: Run['status'];
    positivePrompt: string;
    negativePrompt: string;
    steps: number;
    cfg: number;
    seed: string | null;
    runpodJobId: string | null;
    workerJobDir: string | null;
    durationMs: number | null;
    delayMs: number | null;
    executionMs: number | null;
    errorMessage: string | null;
    rawStatus: unknown;
    startedAt: string | null;
    completedAt: string | null;
    createdAt: string;
    updatedAt: string;
}
export declare function serializeRun(run: Run): RunSerializable;
export interface CreateRunFields {
    positivePrompt: string;
    negativePrompt?: string;
    steps?: number;
    cfg?: number;
    seed?: number;
}
