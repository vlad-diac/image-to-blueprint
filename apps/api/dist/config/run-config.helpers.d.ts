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
export declare function serializeRunConfiguration(row: RunConfiguration): RunConfigurationSerializable;
