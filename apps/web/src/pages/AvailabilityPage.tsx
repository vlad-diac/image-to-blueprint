import { useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  checkAvailability,
  fetchAvailability,
  type AvailabilityDto,
} from '../lib/api';
import {
  assignPoolColors,
  buildAvailabilityChart,
  CHART_INK,
  formatApprox,
  formatWhen,
  MAX_SERIES,
  RANK_BOUNDS,
  type ChartRow,
} from '../lib/availability';

type Props = { onOpenSettings: () => void };

export function AvailabilityPage({ onOpenSettings }: Props) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['availability'],
    queryFn: fetchAvailability,
    staleTime: 30_000,
  });
  const dto = query.data;

  const [selected, setSelected] = useState<string[]>([]);
  const [checking, setChecking] = useState<string[]>([]);
  const initialized = useRef(false);
  const colorRef = useRef<Record<string, string>>({});

  // Seed the list once from pools that already have data.
  if (dto && !initialized.current) {
    initialized.current = true;
    setSelected(dto.pools.map((p) => p.pool).slice(0, MAX_SERIES));
  }

  const colorByPool = useMemo(() => {
    const next = assignPoolColors(selected, colorRef.current);
    colorRef.current = next;
    return next;
  }, [selected]);

  const checkMut = useMutation({
    mutationFn: checkAvailability,
    onMutate: (pools) => setChecking((c) => [...new Set([...c, ...pools])]),
    onSuccess: (fresh: AvailabilityDto) =>
      qc.setQueryData(['availability'], fresh),
    onSettled: (_d, _e, pools) =>
      setChecking((c) => c.filter((p) => !pools.includes(p))),
  });

  const { series, chartData, missing } = useMemo(
    () => buildAvailabilityChart(dto, selected, colorByPool),
    [dto, selected, colorByPool],
  );

  function addPool(pool: string) {
    if (!pool || selected.includes(pool)) return;
    setSelected((s) => [...s, pool]);
    const hasData = dto?.pools.some((p) => p.pool === pool);
    if (!hasData) checkMut.mutate([pool]); // retrieve data for this pool
  }

  function removePool(pool: string) {
    setSelected((s) => s.filter((p) => p !== pool));
  }

  const addable = (dto?.catalog ?? []).filter((p) => !selected.includes(p));
  const atCap = selected.length >= MAX_SERIES;
  const yMax = Math.max(
    12,
    ...chartData.flatMap((row) =>
      series.map((s) => (typeof row[s.pool] === 'number' ? (row[s.pool] as number) : 0)),
    ),
  );

  return (
    <div className="min-h-screen p-6 lg:p-10">
      <div className="mx-auto max-w-6xl space-y-8">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-widest text-sky-500/90">
              Image → Blueprint
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              GPU availability
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Approx GPUs available over time per pool. Add pools to compare — a
              pool with no history is checked live. Higher, steadier lines are the
              safer datacenters to attach a network volume to.
            </p>
          </div>
          <button
            type="button"
            onClick={onOpenSettings}
            className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-200 hover:border-sky-600/70 hover:bg-slate-800/80"
            title="Open sidebar (⌘ D / Ctrl D)"
          >
            <span aria-hidden className="text-lg">
              ⚙
            </span>
            Sidebar
          </button>
        </header>

        {query.isLoading && (
          <p className="text-sm text-slate-400">Loading availability…</p>
        )}
        {query.error && (
          <p className="rounded-lg border border-red-900/50 bg-red-950/35 p-3 text-sm text-red-300">
            {(query.error as Error).message}
          </p>
        )}

        {dto && (
          <>
            {/* Pool selector — chips double as the chart legend. */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-lg font-medium">Pools</h2>
                <span className="text-[11px] text-slate-500">
                  {dto.runs} snapshots · {formatWhen(dto.firstRun)} →{' '}
                  {formatWhen(dto.lastRun)}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {selected.map((pool) => {
                  const isChecking = checking.includes(pool);
                  const noData = missing.includes(pool) && !isChecking;
                  return (
                    <span
                      key={pool}
                      className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/60 py-1 pl-2 pr-1 text-xs"
                    >
                      <span
                        aria-hidden
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ background: colorByPool[pool] }}
                      />
                      <span className="font-mono text-slate-200">{pool}</span>
                      {isChecking && (
                        <span className="text-[10px] text-sky-300">checking…</span>
                      )}
                      {noData && (
                        <span className="text-[10px] text-amber-300">no data</span>
                      )}
                      <button
                        type="button"
                        onClick={() => removePool(pool)}
                        className="rounded-full px-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
                        aria-label={`Remove ${pool}`}
                      >
                        ×
                      </button>
                    </span>
                  );
                })}

                <select
                  className="rounded-full border border-slate-700 bg-slate-950/60 px-3 py-1 text-xs text-slate-200 disabled:opacity-40"
                  value=""
                  disabled={atCap || addable.length === 0}
                  onChange={(e) => {
                    addPool(e.target.value);
                    e.target.value = '';
                  }}
                >
                  <option value="" disabled>
                    {atCap
                      ? `Max ${MAX_SERIES} pools`
                      : addable.length
                        ? '+ Add pool'
                        : 'All pools added'}
                  </option>
                  {addable.map((p) => (
                    <option key={p} value={p}>
                      {p}
                      {dto.pools.some((d) => d.pool === p) ? '' : ' (fetch live)'}
                    </option>
                  ))}
                </select>
              </div>
              {checkMut.error && (
                <p className="mt-3 text-xs text-red-400">
                  {(checkMut.error as Error).message}
                </p>
              )}
            </section>

            {/* The single chart. */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              {series.length === 0 ? (
                <div className="flex min-h-[320px] items-center justify-center text-sm text-slate-500">
                  {checking.length
                    ? 'Fetching current availability…'
                    : 'Add a pool to see its availability over time.'}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 8, right: 64, bottom: 8, left: 8 }}
                  >
                    <CartesianGrid stroke={CHART_INK.grid} vertical={false} />
                    <XAxis
                      dataKey="label"
                      tick={{ fill: CHART_INK.text, fontSize: 11 }}
                      stroke={CHART_INK.axis}
                      minTickGap={24}
                    />
                    <YAxis
                      domain={[0, yMax]}
                      tick={{ fill: CHART_INK.text, fontSize: 11 }}
                      stroke={CHART_INK.axis}
                      width={52}
                      label={{
                        value: 'approx GPUs',
                        angle: -90,
                        position: 'insideLeft',
                        style: {
                          textAnchor: 'middle',
                          fill: CHART_INK.text,
                          fontSize: 11,
                        },
                      }}
                    />
                    {RANK_BOUNDS.map((b) => (
                      <ReferenceLine
                        key={b.name}
                        y={b.at}
                        stroke={CHART_INK.reference}
                        strokeDasharray="3 4"
                        label={{
                          value: b.name,
                          position: 'right',
                          fill: CHART_INK.axis,
                          fontSize: 10,
                        }}
                      />
                    ))}
                    <Tooltip content={<ChartTooltip />} />
                    {series.map((s) => (
                      <Line
                        key={s.pool}
                        type="monotone"
                        dataKey={s.pool}
                        name={s.pool}
                        stroke={s.color}
                        strokeWidth={2}
                        dot={{ r: 2.5, fill: s.color, strokeWidth: 0 }}
                        activeDot={{ r: 4 }}
                        connectNulls
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </section>

            {/* Per-pool readout — dumb render of the util's per-pool data. */}
            {series.length > 0 && (
              <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {series.map((s) => (
                  <div
                    key={s.pool}
                    className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        aria-hidden
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ background: s.color }}
                      />
                      <h3 className="font-mono text-sm text-slate-100">{s.pool}</h3>
                    </div>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                      <Stat
                        label="latest"
                        value={s.latest ? formatApprox(s.latest.approx) : '—'}
                        sub={s.latest?.stock ?? ''}
                      />
                      <Stat label="avg" value={formatApprox(s.avgApprox)} sub="all runs" />
                      <Stat
                        label="uptime"
                        value={`${Math.round(s.availability * 100)}%`}
                        sub="had stock"
                      />
                    </div>
                    <div className="mt-3 border-t border-slate-800 pt-3">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">
                        Top datacenters
                      </div>
                      <ul className="mt-1 space-y-1 text-xs text-slate-300">
                        {s.topRegions.length === 0 && (
                          <li className="text-slate-500">—</li>
                        )}
                        {s.topRegions.map((r) => (
                          <li key={r.dc} className="flex justify-between gap-2">
                            <span className="font-mono">
                              {r.dc}
                              {r.location ? (
                                <span className="text-slate-500"> · {r.location}</span>
                              ) : null}
                            </span>
                            <span className="tabular-nums text-slate-400">
                              {formatApprox(r.avgApprox)} ({Math.round(r.share * 100)}%)
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: Readonly<{ label: string; value: string; sub?: string }>) {
  return (
    <div className="rounded-lg bg-slate-950/50 px-2 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-0.5 text-base font-semibold text-slate-100 tabular-nums">
        {value}
      </div>
      {sub ? <div className="text-[10px] text-slate-500">{sub}</div> : null}
    </div>
  );
}

interface TooltipEntry {
  dataKey?: string | number;
  name?: string | number;
  value?: number;
  color?: string;
  payload?: ChartRow;
}

function ChartTooltip(props: Readonly<{
  active?: boolean;
  payload?: TooltipEntry[];
}>) {
  const { active, payload } = props;
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload as ChartRow | undefined;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950/95 p-3 text-xs shadow-xl">
      <div className="mb-1 font-medium text-slate-200">{row?.label}</div>
      <ul className="space-y-0.5">
        {payload
          .filter((e) => typeof e.value === 'number')
          .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
          .map((e) => (
            <li key={String(e.dataKey)} className="flex items-center gap-2">
              <span
                aria-hidden
                className="h-2 w-2 rounded-full"
                style={{ background: e.color }}
              />
              <span className="font-mono text-slate-300">{e.name}</span>
              <span className="ml-auto tabular-nums text-slate-100">
                {formatApprox(e.value as number)} GPUs
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}
