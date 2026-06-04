"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isTerminalRunStatus = isTerminalRunStatus;
exports.serializeRun = serializeRun;
const client_1 = require("@prisma/client");
const TERMINAL = new Set([
    client_1.RunStatus.SUCCEEDED,
    client_1.RunStatus.FAILED,
    client_1.RunStatus.CANCELLED,
    client_1.RunStatus.TIMED_OUT,
]);
function isTerminalRunStatus(s) {
    return TERMINAL.has(s);
}
function toIso(d) {
    if (!d)
        return null;
    return d.toISOString();
}
function metaRow(o) {
    return {
        index: o.index,
        seed: o.seed !== null ? o.seed.toString() : null,
        width: o.width ?? null,
        height: o.height ?? null,
    };
}
function serializeRun(run) {
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
//# sourceMappingURL=runs.helpers.js.map