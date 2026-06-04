import { type DemoPreset } from './demoDefaults';

export const API_URL =
  import.meta.env.VITE_API_URL ?? 'http://localhost:3001';

export type RunStatus =
  | 'QUEUED'
  | 'IN_QUEUE'
  | 'IN_PROGRESS'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMED_OUT';

export interface OutputMetaDto {
  index: number;
  seed: string | null;
  width: number | null;
  height: number | null;
}

export interface RunDto {
  id: string;
  status: RunStatus;
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
  outputs: OutputMetaDto[];
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface RunConfigDto {
  steps: number;
  cfg: number;
  positivePrompt: string;
  negativePrompt: string;
  numImages: number;
  useLightningLora: boolean;
  updatedAt: string;
}

/** Payload for PUT /config (no updatedAt). */
export type RunConfigUpsertBody = Omit<RunConfigDto, 'updatedAt'>;

export function isPendingStatus(s: RunStatus): boolean {
  return s === 'QUEUED' || s === 'IN_QUEUE' || s === 'IN_PROGRESS';
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function inputUrl(id: string): string {
  return `${API_URL}/runs/${id}/input.png`;
}

export function outputUrl(id: string, index: number): string {
  return `${API_URL}/runs/${id}/outputs/${index}.png`;
}

export function fetchConfig(): Promise<RunConfigDto> {
  return fetchJson<RunConfigDto>('/config');
}

export function saveConfig(body: RunConfigUpsertBody): Promise<RunConfigDto> {
  return fetchJson<RunConfigDto>('/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** Build FormData for POST /runs from a preset (Demo or advanced). */
export function runFormData(image: File, preset: DemoPreset): FormData {
  const fd = new FormData();
  fd.append('image', image);
  fd.append('positivePrompt', preset.positivePrompt.trim());
  fd.append('negativePrompt', preset.negativePrompt ?? '');
  fd.append('steps', String(preset.steps));
  fd.append('cfg', String(preset.cfg));
  if (preset.seed !== null && preset.seed !== undefined) {
    fd.append('seed', String(preset.seed));
  }
  fd.append('numImages', String(preset.numImages));
  return fd;
}
