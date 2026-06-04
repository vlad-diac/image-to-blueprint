import { useEffect, useMemo, useRef, useState } from 'react';
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import {
  fetchJson,
  inputUrl,
  isPendingStatus,
  outputUrl,
  runFormData,
  saveConfig,
  type RunConfigDto,
  type RunDto,
  type RunStatus,
} from './lib/api';
import { type DemoPreset, DEMO_DEFAULTS } from './lib/demoDefaults';

type MainLayoutProps = {
  demoMode: boolean;
  onOpenSettings: () => void;
  config?: RunConfigDto;
};

export function MainLayout({
  demoMode,
  onOpenSettings,
  config,
}: MainLayoutProps) {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const [positive, setPositive] = useState(DEMO_DEFAULTS.positivePrompt);
  const [negative, setNegative] = useState(DEMO_DEFAULTS.negativePrompt);
  const [steps, setSteps] = useState(DEMO_DEFAULTS.steps);
  const [cfg, setCfg] = useState(DEMO_DEFAULTS.cfg);
  const [seed, setSeed] = useState<string>(
    DEMO_DEFAULTS.seed != null ? String(DEMO_DEFAULTS.seed) : '',
  );
  const [numImages, setNumImages] = useState(DEMO_DEFAULTS.numImages);

  const [lightbox, setLightbox] = useState<number | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [useLightningLora, setUseLightningLora] = useState(true);
  const lastRequestedNumImages = useRef(DEMO_DEFAULTS.numImages);
  const syncedFromConfig = useRef(false);

  const effectiveDemoPreset = useMemo((): DemoPreset => {
    if (!config) return DEMO_DEFAULTS;
    return {
      ...DEMO_DEFAULTS,
      positivePrompt: config.positivePrompt,
      negativePrompt: config.negativePrompt,
      steps: config.steps,
      cfg: config.cfg,
      numImages: config.numImages,
      seed: null,
    };
  }, [config]);

  useEffect(() => {
    if (!config || syncedFromConfig.current) return;
    syncedFromConfig.current = true;
    setPositive(config.positivePrompt);
    setNegative(config.negativePrompt);
    setSteps(config.steps);
    setCfg(config.cfg);
    setNumImages(config.numImages);
    setUseLightningLora(config.useLightningLora);
  }, [config]);

  const advancedPreset = useMemo(
    (): DemoPreset => ({
      positivePrompt: positive,
      negativePrompt: negative,
      steps,
      cfg,
      seed:
        seed.trim() === ''
          ? null
          : Number.isFinite(Number(seed))
            ? Number(seed)
            : null,
      numImages: Math.min(Math.max(numImages, 1), 4),
    }),
    [positive, negative, steps, cfg, seed, numImages],
  );

  const listQuery = useQuery({
    queryKey: ['runs'],
    queryFn: () => fetchJson<RunDto[]>('/runs?limit=40'),
    refetchInterval: 5000,
    enabled: !demoMode,
  });

  const runQuery = useQuery({
    queryKey: ['run', selectedId],
    queryFn: () => fetchJson<RunDto>(`/runs/${selectedId}`),
    enabled: Boolean(selectedId),
    placeholderData: keepPreviousData,
    refetchInterval: (q) => {
      const d = q.state.data;
      return d?.status && isPendingStatus(d.status) ? 1500 : false;
    },
  });

  const createMut = useMutation({
    mutationFn: async (args: { image: File; preset: DemoPreset }) => {
      if (!args.preset.positivePrompt.trim()) {
        throw new Error('Positive prompt is required.');
      }
      const body = runFormData(args.image, args.preset);
      return fetchJson<RunDto>('/runs', { method: 'POST', body });
    },
    onSuccess: (r) => {
      void qc.invalidateQueries({ queryKey: ['runs'] });
      setSelectedId(r.id);
    },
  });

  const cancelMut = useMutation({
    mutationFn: async () => {
      if (!selectedId) return;
      return fetchJson<RunDto>(`/runs/${selectedId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['runs'] });
      if (selectedId) {
        await qc.invalidateQueries({ queryKey: ['run', selectedId] });
      }
    },
  });

  const saveConfigMut = useMutation({
    mutationFn: saveConfig,
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['config'] });
    },
  });

  function runWithPreset(img: File, preset: DemoPreset) {
    lastRequestedNumImages.current = preset.numImages;
    createMut.mutate({ image: img, preset });
  }

  function onPickFile(next: File | null) {
    setFile(next);
    setSelectedId(null);

    if (!next) return;

    lastRequestedNumImages.current = demoMode
      ? effectiveDemoPreset.numImages
      : advancedPreset.numImages;

    if (demoMode) runWithPreset(next, effectiveDemoPreset);
  }

  const selected = runQuery.data;
  const busy =
    createMut.isPending || cancelMut.isPending || saveConfigMut.isPending;

  const tileCount = useMemo(() => {
    if (selected?.numImages != null) return selected.numImages;
    return lastRequestedNumImages.current;
  }, [selected?.numImages, selected?.id]);

  function submitAdvanced() {
    if (!file) throw new Error('Choose an image first.');
    runWithPreset(file, advancedPreset);
  }

  function statusLabel(status: RunStatus | undefined): string {
    if (!status) return 'Waiting';
    if (status === 'IN_QUEUE') return 'Queued…';
    if (status === 'IN_PROGRESS') return 'Rendering…';
    if (status === 'QUEUED') return 'Submitted…';
    return status;
  }

  const showConfig = !demoMode;

  return (
    <div className="min-h-screen p-6 lg:p-10">
      {/* Lightbox */}
      {lightbox !== null && selectedId ? (
        <div
          role="presentation"
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 p-4"
          onClick={() => setLightbox(null)}
        >
          <button
            type="button"
            className="absolute right-6 top-6 rounded-lg bg-slate-800 px-3 py-1 text-sm text-slate-100"
            onClick={() => setLightbox(null)}
          >
            Close
          </button>
          <div role="presentation" onClick={(e) => e.stopPropagation()}>
            <img
              src={outputUrl(selectedId, lightbox)}
              alt={`Output variant ${lightbox}`}
              className="max-h-[90vh] max-w-[94vw] rounded-lg border border-slate-700 shadow-2xl"
            />
          </div>
        </div>
      ) : null}

      <div className="mx-auto max-w-6xl space-y-8">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-widest text-sky-500/90">
              Image → Blueprint
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              Naval recognition sketches
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              {demoMode
                ? 'Demo: drop or pick one photo — we run with the blueprint preset.'
                : 'Configure prompts below, then process. Use the sidebar switch for demo mode.'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onOpenSettings()}
            className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-200 hover:border-sky-600/70 hover:bg-slate-800/80"
            title="Open settings (⌘ D / Ctrl D)"
          >
            <span aria-hidden className="text-lg">
              ⚙
            </span>
            Sidebar
          </button>
        </header>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* INPUT */}
          <section
            className={`relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/55 p-5 ${dragActive ? 'ring-2 ring-sky-500/70' : ''}`}
            onDragEnter={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              if (e.currentTarget === e.target) setDragActive(false);
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              const dropped = e.dataTransfer.files?.[0];
              if (dropped && dropped.type.startsWith('image/')) onPickFile(dropped);
            }}
          >
            <h2 className="mb-4 text-lg font-medium">Reference photo</h2>
            <input
              id="inp-file"
              className="sr-only"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
            />
            <label
              htmlFor="inp-file"
              className="flex min-h-[240px] cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/50 px-6 py-10 hover:border-sky-600/50"
            >
              {file ? (
                <img
                  src={URL.createObjectURL(file)}
                  alt="Chosen input preview"
                  className="max-h-[320px] w-full rounded-lg object-contain shadow-inner ring-1 ring-slate-800"
                />
              ) : (
                <>
                  <p className="text-center text-sm text-slate-400">
                    Drag &amp; drop a vessel photo here,{' '}
                    <span className="text-sky-400 underline"> or browse files</span>
                  </p>
                  <span className="rounded-full bg-sky-950/70 px-3 py-1 text-[11px] font-mono text-sky-300">
                    PNG / JPEG / WebP
                  </span>
                </>
              )}
            </label>

            {!demoMode && (
              <p className="mt-4 text-[11px] text-slate-500">
                In advanced mode choose an image above, tweak settings underneath, then
                click <strong>Process</strong>.
              </p>
            )}
          </section>

          {/* OUTPUT GALLERY */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-medium">Variants</h2>
              <div className="flex flex-wrap items-center gap-2">
                {selected?.status ? (
                  <span className="rounded-full bg-slate-950 px-3 py-1 text-[11px] font-mono text-sky-300 ring-1 ring-slate-800">
                    {statusLabel(selected.status)}
                  </span>
                ) : null}
                {selectedId &&
                selected &&
                isPendingStatus(selected.status as RunStatus) ? (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void cancelMut.mutateAsync()}
                    className="rounded-lg border border-amber-900/60 px-3 py-1 text-[11px] text-amber-200 hover:bg-amber-950/50 disabled:opacity-40"
                  >
                    Cancel run
                  </button>
                ) : null}
              </div>
            </div>

            {!selectedId && !busy && (
              <div className="flex min-h-[240px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/35 text-center text-sm text-slate-500">
                Output gallery appears here after your first run completes.
              </div>
            )}

            {(selectedId || busy) && (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {Array.from({ length: tileCount }).map((_, i) => {
                  const succeeded = selected?.status === 'SUCCEEDED';
                  const showPending =
                    (busy && !selectedId) ||
                    (selected != null &&
                      isPendingStatus(selected.status as RunStatus));
                  const key = `${selectedId ?? 'queued'}-${i}`;
                  const sid = selectedId;

                  if (showPending) {
                    return (
                      <div
                        key={key}
                        className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/80 pb-[100%]"
                      >
                        <div className="absolute inset-[10%]">
                          <div className="h-full rounded-lg bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 shimmer" />
                          <span className="absolute bottom-[-1.75rem] left-0 text-[10px] font-mono text-slate-500">
                            slot {i + 1}
                          </span>
                        </div>
                      </div>
                    );
                  }

                  if (succeeded && sid) {
                    const sn = selected.outputs[i]?.seed ?? '';
                    return (
                      <figure
                        key={key}
                        className="group relative rounded-xl border border-slate-800 bg-slate-950/40 p-2"
                      >
                        <button
                          type="button"
                          className="block w-full"
                          onClick={() => setLightbox(i)}
                        >
                          <img
                            src={outputUrl(sid, i)}
                            alt={`Blueprint variant ${i + 1}`}
                            className="w-full rounded-lg object-contain"
                          />
                        </button>
                        <figcaption className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
                          <a
                            className="rounded bg-slate-800 px-2 py-0.5 text-sky-300 hover:bg-slate-700"
                            href={outputUrl(sid, i)}
                            download={`blueprint_${sid}_${i}.png`}
                          >
                            save
                          </a>
                          {sn ? (
                            <span className="font-mono opacity-75">
                              seed {sn.slice(0, 8)}…
                            </span>
                          ) : null}
                        </figcaption>
                      </figure>
                    );
                  }

                  return (
                    <div
                      key={key}
                      className="flex min-h-[120px] items-center justify-center rounded-xl border border-slate-800 bg-slate-950/70 text-[11px] text-slate-500"
                    >
                      —
                    </div>
                  );
                })}
              </div>
            )}

            {(selected?.status === 'FAILED' ||
              selected?.status === 'TIMED_OUT' ||
              selected?.status === 'CANCELLED') && (
              <p className="mt-4 rounded-lg border border-red-900/50 bg-red-950/35 p-3 text-sm text-red-300">
                {selected.errorMessage ?? selected.status}{' '}
                <button
                  type="button"
                  className="ml-2 text-sky-400 underline"
                  onClick={() =>
                    file && demoMode && runWithPreset(file, effectiveDemoPreset)
                  }
                >
                  retry (demo preset)
                </button>
              </p>
            )}
          </section>
        </div>

        {/* Advanced configuration */}
        {showConfig ? (
          <div className="grid gap-6 lg:grid-cols-[1.05fr_.95fr]">
            <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
              <h2 className="text-lg font-medium">Inference settings</h2>
              <label className="block text-sm text-slate-300">
                Positive prompt
                <textarea
                  className="mt-1 w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-sm outline-none ring-sky-500 focus:ring-2"
                  rows={8}
                  value={positive}
                  onChange={(e) => setPositive(e.target.value)}
                />
              </label>
              <label className="block text-sm text-slate-300">
                Negative prompt
                <textarea
                  className="mt-1 w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-2 text-sm outline-none ring-sky-500 focus:ring-2"
                  rows={2}
                  value={negative}
                  onChange={(e) => setNegative(e.target.value)}
                />
              </label>

              <div className="flex flex-wrap gap-4">
                <label className="text-sm text-slate-300">
                  Steps
                  <input
                    type="number"
                    min={1}
                    max={100}
                    className="mt-1 block w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-sm"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                  />
                </label>
                <label className="text-sm text-slate-300">
                  CFG
                  <input
                    type="number"
                    step={0.1}
                    className="mt-1 block w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-sm"
                    value={cfg}
                    onChange={(e) => setCfg(Number(e.target.value))}
                  />
                </label>
                <label className="text-sm text-slate-300">
                  Seed{' '}
                  <span className="text-[10px] text-slate-500">(leave empty → worker random bases)</span>
                  <input
                    type="text"
                    inputMode="numeric"
                    placeholder="random"
                    className="mt-1 block w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 font-mono text-sm"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </label>
                <label className="text-sm text-slate-300">
                  Number of images
                  <input
                    type="number"
                    min={1}
                    max={4}
                    className="mt-1 block w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-sm"
                    value={numImages}
                    onChange={(e) =>
                      setNumImages(
                        Math.min(4, Math.max(1, Number(e.target.value) || 1)),
                      )
                    }
                  />
                </label>
              </div>

              <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                <input
                  type="checkbox"
                  className="h-5 w-5 shrink-0 accent-sky-500"
                  checked={useLightningLora}
                  onChange={(e) => setUseLightningLora(e.target.checked)}
                />
                <span>
                  Use Lightning LoRA{' '}
                  <span className="text-[10px] text-slate-500">
                    (saved with preset; not sent to RunPod yet)
                  </span>
                </span>
              </label>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={busy || !file}
                  onClick={() => void Promise.resolve(submitAdvanced()).catch(() => {})}
                  className="rounded-xl bg-sky-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
                >
                  Process
                </button>
                <button
                  type="button"
                  disabled={
                    busy || !positive.trim() || saveConfigMut.isPending
                  }
                  onClick={() => {
                    saveConfigMut.mutate({
                      positivePrompt: positive.trim(),
                      negativePrompt: negative,
                      steps,
                      cfg,
                      numImages: Math.min(Math.max(numImages, 1), 4),
                      useLightningLora,
                    });
                  }}
                  className="rounded-xl border border-slate-600 bg-slate-800/80 px-5 py-2.5 text-sm font-medium text-slate-100 hover:bg-slate-700/80 disabled:opacity-50"
                >
                  Save config
                </button>
              </div>
              {saveConfigMut.error && (
                <p className="text-sm text-red-400">
                  {(saveConfigMut.error as Error).message}
                </p>
              )}
              {createMut.error && (
                <p className="text-sm text-red-400">
                  {(createMut.error as Error).message}
                </p>
              )}
            </section>

            <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
              <h2 className="text-lg font-medium">Recent runs</h2>
              <div className="max-h-[420px] overflow-auto rounded-xl border border-slate-800">
                <table className="min-w-full text-left text-xs text-slate-300">
                  <thead className="sticky top-0 bg-slate-950/95 text-[10px] uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-2 py-2">Thumb</th>
                      <th className="px-2 py-2">Status</th>
                      <th className="px-2 py-2">Id</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(listQuery.data ?? []).map((r) => (
                      <tr
                        key={r.id}
                        className={
                          selectedId === r.id
                            ? 'cursor-pointer bg-sky-900/25'
                            : 'cursor-pointer odd:bg-slate-950/30 hover:bg-slate-800/40'
                        }
                        onClick={() => {
                          setSelectedId(r.id);
                          // keep file as-is; user can still see output
                        }}
                      >
                        <td className="px-2 py-1">
                          {r.status === 'SUCCEEDED' ? (
                            <img
                              src={outputUrl(r.id, 0)}
                              alt=""
                              className="h-10 w-14 rounded object-cover ring-1 ring-slate-800"
                            />
                          ) : (
                            <div className="h-10 w-14 rounded bg-slate-800" />
                          )}
                        </td>
                        <td className="px-2 py-1 font-mono text-[10px]">{r.status}</td>
                        <td className="px-2 py-1 font-mono text-[10px]" title={r.id}>
                          {r.id.slice(0, 8)}…
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        ) : null}

        {/* Advanced only: stored input + JSON (hidden in Demo mode). */}
        {showConfig && selectedId && (
          <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-medium">Run details</h2>
              {selected?.status ? (
                <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-mono">
                  {selected.status}
                </span>
              ) : null}
            </div>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div>
                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                  Stored input (API)
                </div>
                <img
                  src={inputUrl(selectedId)}
                  alt="Server input"
                  className="mt-2 max-h-[200px] w-full rounded-lg border border-slate-800 object-contain bg-slate-950"
                />
              </div>
            </div>
            <details className="group mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-sm">
              <summary className="cursor-pointer text-slate-200">JSON</summary>
              <div className="mt-3 space-y-2 text-xs text-slate-300">
                {runQuery.isLoading && <p>Loading…</p>}
                {runQuery.error && (
                  <p className="text-red-400">
                    {(runQuery.error as Error).message}
                  </p>
                )}
                {selected && (
                  <pre className="max-h-[360px] overflow-auto rounded-lg bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-200">
                    {JSON.stringify(selected, null, 2)}
                  </pre>
                )}
              </div>
            </details>
          </section>
        )}
      </div>
    </div>
  );
}
