import type { RunConfigDto } from '../lib/api';

type SidebarProps = {
  open: boolean;
  onClose: () => void;
  demoMode: boolean;
  onDemoChange: (v: boolean) => void;
  config?: RunConfigDto;
  configLoading?: boolean;
  configError?: string;
};

function formatUpdatedAt(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function Sidebar({
  open,
  onClose,
  demoMode,
  onDemoChange,
  config,
  configLoading,
  configError,
}: SidebarProps) {
  if (!open) return null;

  return (
    <>
      <button
        type="button"
        aria-label="Close sidebar"
        className="fixed inset-0 z-40 cursor-default bg-black/55 backdrop-blur-[2px]"
        onClick={onClose}
      />
      <aside
        className="fixed left-0 top-0 z-50 flex h-full w-[min(20rem,calc(100vw-3rem))] flex-col gap-6 border-r border-slate-800 bg-slate-950/98 p-5 shadow-2xl transition-transform duration-300"
        aria-label="Settings"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">Quick settings</h2>
            <p className="mt-1 text-xs text-slate-500">
              Toggle opens with ⌘ D (Mac) or Ctrl D.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            Close
          </button>
        </div>

        <label className="flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/80 p-4">
          <div>
            <div className="text-sm font-medium text-slate-200">Demo mode</div>
            <div className="mt-1 text-[11px] leading-snug text-slate-500">
              Only input + outputs. Drops use the blueprint preset automatically.
              Turn off for prompts &amp; number of variants.
            </div>
          </div>
          <input
            type="checkbox"
            className="h-5 w-5 accent-sky-500"
            checked={demoMode}
            onChange={(e) => onDemoChange(e.target.checked)}
          />
        </label>

        <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-900/55 p-4">
          <div className="text-sm font-semibold text-slate-200">Saved preset</div>
          <p className="text-[11px] leading-snug text-slate-500">
            Read-only defaults from the server (used for demo drops when loaded).
          </p>
          {configLoading && (
            <p className="text-xs text-slate-500">Loading…</p>
          )}
          {configError && (
            <p className="text-xs text-amber-400/90">{configError}</p>
          )}
          {config && (
            <div className="mt-2 space-y-2 text-[11px] text-slate-300">
              <div className="flex flex-wrap gap-2">
                <span className="rounded bg-slate-800 px-2 py-0.5 font-mono">
                  steps {config.steps}
                </span>
                <span className="rounded bg-slate-800 px-2 py-0.5 font-mono">
                  cfg {config.cfg}
                </span>
                <span className="rounded bg-slate-800 px-2 py-0.5 font-mono">
                  n {config.numImages}
                </span>
                <span
                  className={`rounded px-2 py-0.5 font-mono ${config.useLightningLora ? 'bg-emerald-950/70 text-emerald-200' : 'bg-slate-800 text-slate-400'}`}
                >
                  lightning {config.useLightningLora ? 'on' : 'off'}
                </span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-wide text-slate-500">
                  Positive prompt
                </span>
                <p className="mt-1 whitespace-pre-wrap break-words leading-snug text-slate-400">
                  {config.positivePrompt.length > 120
                    ? `${config.positivePrompt.slice(0, 120)}\u2026`
                    : config.positivePrompt}
                </p>
              </div>
              <div className="text-[10px] text-slate-500">
                Last saved: {formatUpdatedAt(config.updatedAt)}
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
