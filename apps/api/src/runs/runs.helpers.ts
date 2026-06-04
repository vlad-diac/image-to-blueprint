import { RunStatus, type Run, type RunImage } from '@prisma/client';

const TERMINAL = new Set<RunStatus>([
  RunStatus.SUCCEEDED,
  RunStatus.FAILED,
  RunStatus.CANCELLED,
  RunStatus.TIMED_OUT,
]);

export function isTerminalRunStatus(s: RunStatus): boolean {
  return TERMINAL.has(s);
}

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

function toIso(d: Date | null): string | null {
  if (!d) return null;
  return d.toISOString();
}

function metaRow(o: RunOutputsMetaRow) {
  return {
    index: o.index,
    seed: o.seed !== null ? o.seed.toString() : null,
    width: o.width ?? null,
    height: o.height ?? null,
  };
}

/** Run serialized for API responses (omit input bytes unless loading full entity). */
export function serializeRun(run: SerializedRunPayload): RunSerializable {
  const outs = [...(run.outputs ?? [])].sort((a, b) => a.index - b.index);
  return {
    id: run.id,
    status: run.status,
    positivePrompt: run.positivePrompt,
    negativePrompt: run.negativePrompt,
    steps: run.steps,
    cfg: run.cfg,
    numImages: run.numImages,
    seed: run.seed !== null ? run.seed.toString() : null,
    runpodJobId: run.runpodJobId,
    workerJobDir: run.workerJobDir,
    durationMs: run.durationMs ?? null,
    delayMs: run.delayMs ?? null,
    executionMs: run.executionMs ?? null,
    errorMessage: run.errorMessage,
    rawStatus: run.rawStatus,
    outputs: outs.map(metaRow),
    startedAt: toIso(run.startedAt),
    completedAt: toIso(run.completedAt),
    createdAt: run.createdAt.toISOString(),
    updatedAt: run.updatedAt.toISOString(),
  };
}

export interface CreateRunFields {
  positivePrompt: string;
  negativePrompt?: string;
  steps?: number;
  cfg?: number;
  seed?: number;
  /** 1–4 (default 3). */
  numImages?: number;
}
