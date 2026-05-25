import type { Run, RunStatus } from '@prisma/client';

const TERMINAL = new Set<RunStatus>([
  RunStatus.SUCCEEDED,
  RunStatus.FAILED,
  RunStatus.CANCELLED,
  RunStatus.TIMED_OUT,
]);

export function isTerminalRunStatus(s: RunStatus): boolean {
  return TERMINAL.has(s);
}

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

function toIso(d: Date | null): string | null {
  if (!d) return null;
  return d.toISOString();
}

export function serializeRun(run: Run): RunSerializable {
  return {
    id: run.id,
    status: run.status,
    positivePrompt: run.positivePrompt,
    negativePrompt: run.negativePrompt,
    steps: run.steps,
    cfg: run.cfg,
    seed: run.seed !== null ? run.seed.toString() : null,
    runpodJobId: run.runpodJobId,
    workerJobDir: run.workerJobDir,
    durationMs: run.durationMs ?? null,
    delayMs: run.delayMs ?? null,
    executionMs: run.executionMs ?? null,
    errorMessage: run.errorMessage,
    rawStatus: run.rawStatus,
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
}
