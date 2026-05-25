import { useMemo, useState } from 'react';
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:3001';

export type RunStatus =
  | 'QUEUED'
  | 'IN_QUEUE'
  | 'IN_PROGRESS'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMED_OUT';

export interface RunDto {
  id: string;
  status: RunStatus;
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

function isPendingStatus(s: RunStatus): boolean {
  return s === 'QUEUED' || s === 'IN_QUEUE' || s === 'IN_PROGRESS';
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export default function App() {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [positive, setPositive] = useState('');
  const [negative, setNegative] = useState('');
  const [steps, setSteps] = useState(4);
  const [cfg, setCfg] = useState(1);
  const [seed, setSeed] = useState<string>('');

  const listQuery = useQuery({
    queryKey: ['runs'],
    queryFn: () => fetchJson<RunDto[]>('/runs?limit=40'),
    refetchInterval: 5000,
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
    mutationFn: async () => {
      if (!file) throw new Error('Choose an image first.');
      if (!positive.trim()) throw new Error('Positive prompt is required.');
      const fd = new FormData();
      fd.append('image', file);
      fd.append('positivePrompt', positive.trim());
      fd.append('negativePrompt', negative);
      fd.append('steps', String(steps));
      fd.append('cfg', String(cfg));
      if (seed.trim() !== '') fd.append('seed', seed.trim());
      return fetchJson<RunDto>('/runs', { method: 'POST', body: fd });
    },
    onSuccess: (r) => {
      void qc.invalidateQueries({ queryKey: ['runs'] });
      setSelectedId(r.id);
    },
  });

  const selected = runQuery.data;
  const previewUrls = useMemo(() => {
    if (!selectedId) return null;
    return {
      in: `${API_URL}/runs/${selectedId}/input.png`,
      out: `${API_URL}/runs/${selectedId}/output.png`,
    };
  }, [selectedId]);

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

  const busy = createMut.isPending || cancelMut.isPending;

  return (
    <div className="min-h-screen p-6 lg:p-10">
      <div className="mx-auto max-w-6xl space-y-8">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">Image → Blueprint</h1>
          <p className="mt-1 text-sm text-slate-400">
            POC: uploads trigger a RunPod Serverless pipeline; statuses refresh via Nest (
            <code className="text-slate-300">/run</code> + <code className="text-slate-300">/status</code>
            ).
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.1fr_.9fr]">
          <section className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
            <h2 className="text-lg font-medium">Run</h2>
            <label className="block text-sm text-slate-300">
              Image
              <input
                className="mt-1 block w-full text-sm text-slate-300 file:mr-3 file:rounded-md file:border-0 file:bg-slate-700 file:px-3 file:py-1.5 file:text-sm file:text-slate-100"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="block text-sm text-slate-300">
              Positive prompt
              <textarea
                className="mt-1 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm outline-none ring-sky-500 focus:ring-2"
                rows={8}
                value={positive}
                onChange={(e) => setPositive(e.target.value)}
                placeholder="<sks> front view ..."
              />
            </label>
            <label className="block text-sm text-slate-300">
              Negative prompt
              <textarea
                className="mt-1 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm outline-none ring-sky-500 focus:ring-2"
                rows={2}
                value={negative}
                onChange={(e) => setNegative(e.target.value)}
              />
            </label>
            <div className="flex flex-wrap gap-3">
              <label className="text-sm text-slate-300">
                Steps
                <input
                  type="number"
                  min={1}
                  max={100}
                  className="ml-2 w-24 rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-sm"
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                />
              </label>
              <label className="text-sm text-slate-300">
                CFG
                <input
                  type="number"
                  step={0.1}
                  className="ml-2 w-24 rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-sm"
                  value={cfg}
                  onChange={(e) => setCfg(Number(e.target.value))}
                />
              </label>
              <label className="text-sm text-slate-300">
                Seed <span className="text-xs text-slate-500">(optional)</span>
                <input
                  type="number"
                  className="ml-2 w-32 rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-sm"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                disabled={busy}
                onClick={() => void createMut.mutateAsync()}
                className="rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
              >
                Process
              </button>
              <button
                type="button"
                disabled={
                  !selectedId ||
                  busy ||
                  !selected ||
                  selected.status === 'SUCCEEDED' ||
                  selected.status === 'FAILED' ||
                  selected.status === 'CANCELLED' ||
                  selected.status === 'TIMED_OUT'
                }
                onClick={() => void cancelMut.mutateAsync()}
                className="rounded-md bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
              >
                Cancel run
              </button>
            </div>
            {createMut.error && (
              <p className="text-sm text-red-400">
                {(createMut.error as Error).message ?? String(createMut.error)}
              </p>
            )}
          </section>

          <section className="space-y-3 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
            <h2 className="text-lg font-medium">Recent runs</h2>
            <div className="max-h-[360px] overflow-auto rounded-lg border border-slate-800">
              <table className="min-w-full text-left text-xs text-slate-300">
                <thead className="sticky top-0 bg-slate-950/95 text-[11px] uppercase tracking-wide text-slate-400">
                  <tr>
                    <th className="px-3 py-2 font-medium">Status</th>
                    <th className="px-3 py-2 font-medium">Id</th>
                    <th className="px-3 py-2 font-medium">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {(listQuery.data ?? []).map((r) => (
                    <tr
                      key={r.id}
                      className={
                        selectedId === r.id
                          ? 'cursor-pointer bg-sky-900/30 hover:bg-sky-900/40'
                          : 'cursor-pointer odd:bg-slate-950/40 hover:bg-slate-800/50'
                      }
                      onClick={() => setSelectedId(r.id)}
                    >
                      <td className="whitespace-nowrap px-3 py-1.5 font-mono">{r.status}</td>
                      <td className="px-3 py-1.5 font-mono text-[11px]" title={r.id}>
                        {r.id.slice(0, 8)}…
                      </td>
                      <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[11px]">
                        {new Date(r.updatedAt).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {listQuery.isLoading && (
              <p className="text-xs text-slate-500">Loading runs…</p>
            )}
            {listQuery.error && (
              <p className="text-xs text-red-400">{String((listQuery.error as Error).message)}</p>
            )}
          </section>
        </div>

        <section className="space-y-3 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-lg font-medium">Preview</h2>
            {selected?.status ? (
              <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-mono">
                {selected.status}
              </span>
            ) : null}
          </div>

          {!selectedId && (
            <p className="text-sm text-slate-400">Pick a row or submit a run to see previews.</p>
          )}
          {selectedId && previewUrls && (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wide text-slate-400">Input</div>
                <img
                  src={previewUrls.in}
                  alt="Input"
                  className="max-h-[420px] w-full rounded-lg border border-slate-800 object-contain bg-slate-950"
                />
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wide text-slate-400">Output</div>
                {selected?.status === 'SUCCEEDED' ? (
                  <img
                    src={previewUrls.out}
                    alt="Output"
                    className="max-h-[420px] w-full rounded-lg border border-slate-800 object-contain bg-slate-950"
                  />
                ) : (
                  <div className="flex h-[280px] items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950 text-sm text-slate-500">
                    {isPendingStatus(selected?.status ?? 'QUEUED')
                      ? 'Rendering… (RunPod /status polling via Nest)'
                      : 'No output yet'}
                  </div>
                )}
              </div>
            </div>
          )}

          {selectedId && (
            <details className="group rounded-lg border border-slate-800 bg-slate-950/40 p-3 text-sm">
              <summary className="cursor-pointer text-slate-200">Run details (expand)</summary>
              <div className="mt-3 space-y-2 text-xs text-slate-300">
                {runQuery.isLoading && <p>Loading details…</p>}
                {runQuery.error && (
                  <p className="text-red-400">
                    {(runQuery.error as Error).message ?? String(runQuery.error)}
                  </p>
                )}
                {selected && (
                  <pre className="max-h-[360px] overflow-auto rounded-md bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-200">
                    {JSON.stringify(selected, null, 2)}
                  </pre>
                )}
              </div>
            </details>
          )}
        </section>
      </div>
    </div>
  );
}
