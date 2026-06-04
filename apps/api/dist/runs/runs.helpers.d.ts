import { RunStatus, type Run, type RunImage } from '@prisma/client';
export declare function isTerminalRunStatus(s: RunStatus): boolean;
export type RunOutputsMetaRow = Pick<RunImage, 'index' | 'seed' | 'width' | 'height'>;
export interface RunSerializable {
    id: string;
    status: Run['status'];
    positivePrompt: string;
    negativePrompt: string;
    steps: number;
    cfg: number;
    numImages: number;
    seed: string | null;
    runpodJobId: string | null;
    workerJobDir: string | null;
    durationMs: number | null;
    delayMs: number | null;
    executionMs: number | null;
    errorMessage: string | null;
    rawStatus: unknown;
    outputs: Array<{
        index: number;
        seed: string | null;
        width: number | null;
        height: number | null;
    }>;
    startedAt: string | null;
    completedAt: string | null;
    createdAt: string;
    updatedAt: string;
}
export type SerializedRunPayload = Omit<Run, 'inputImage'> & {
    outputs?: RunOutputsMetaRow[];
};
export declare function serializeRun(run: SerializedRunPayload): RunSerializable;
export interface CreateRunFields {
    positivePrompt: string;
    negativePrompt?: string;
    steps?: number;
    cfg?: number;
    seed?: number;
    numImages?: number;
}
