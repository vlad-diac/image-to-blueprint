"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.serializeRunConfiguration = serializeRunConfiguration;
function serializeRunConfiguration(row) {
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
//# sourceMappingURL=run-config.helpers.js.map