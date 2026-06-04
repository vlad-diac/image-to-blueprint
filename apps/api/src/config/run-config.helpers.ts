import type { RunConfiguration } from '@prisma/client';

export interface RunConfigurationSerializable {
  steps: number;
  cfg: number;
  positivePrompt: string;
  negativePrompt: string;
  numImages: number;
  useLightningLora: boolean;
  updatedAt: string;
}

export function serializeRunConfiguration(
  row: RunConfiguration,
): RunConfigurationSerializable {
  return {
    steps: row.steps,
    cfg: row.cfg,
    positivePrompt: row.positivePrompt,
    negativePrompt: row.negativePrompt ?? '',
    numImages: row.numImages,
    useLightningLora: row.useLightningLora,
    updatedAt: row.updatedAt.toISOString(),
  };
}
