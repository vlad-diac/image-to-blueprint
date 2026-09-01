/**
 * Availability view-model utils.
 *
 * Turns the raw `/availability` result into exactly what the chart renders:
 * one merged time-series dataset for a chosen list of pools, plus per-pool
 * series metadata (color, latest reading, stats). View components stay dumb —
 * they take these outputs and render.
 */
import type { AvailabilityDto, PoolStat } from './api';

/**
 * Categorical series palette — the dataviz reference dark theme, validated
 * against the app surface (#0f172a): worst adjacent CVD ΔE 8.4, normal-vision
 * 19.3. Assigned in fixed order, never cycled.
 */
export const POOL_PALETTE = [
  '#3987e5', // blue
  '#d95926', // orange
  '#199e70', // aqua
  '#c98500', // yellow
  '#d55181', // magenta
  '#008300', // green
  '#9085e9', // violet
  '#e66767', // red
] as const;

export const MAX_SERIES = POOL_PALETTE.length;

/** Chart chrome (dark surface). */
export const CHART_INK = {
  grid: '#1e293b', // slate-800
  axis: '#64748b', // slate-500
  text: '#94a3b8', // slate-400
  reference: '#334155', // slate-700
} as const;

/** y-axis reference lines at each rank's lower bound (approx GPUs). */
export const RANK_BOUNDS = [
  { name: 'Low', at: 1 },
  { name: 'Medium', at: 5 },
  { name: 'High', at: 10 },
] as const;

export interface PoolSeries {
  pool: string;
  color: string;
  samples: number;
  avgApprox: number;
  availability: number;
  latest: PoolStat['latest'];
  bestEver: PoolStat['bestEver'];
  topRegions: PoolStat['regions'];
}

export type ChartRow = { t: string; label: string } & Record<
  string,
  number | string | null
>;

export interface AvailabilityChart {
  series: PoolSeries[];
  chartData: ChartRow[];
  /** Selected pools that have no snapshots yet (need an on-demand check). */
  missing: string[];
}

/**
 * Stable pool → color map. Keeps existing assignments (so removing one pool
 * never repaints the survivors) and gives each new pool the lowest free slot.
 */
export function assignPoolColors(
  pools: string[],
  prev: Record<string, string> = {},
): Record<string, string> {
  const next: Record<string, string> = {};
  for (const p of pools) if (prev[p]) next[p] = prev[p];
  const used = new Set(Object.values(next));
  for (const p of pools) {
    if (next[p]) continue;
    const free = POOL_PALETTE.find((c) => !used.has(c));
    next[p] = free ?? POOL_PALETTE[Object.keys(next).length % POOL_PALETTE.length];
    used.add(next[p]);
  }
  return next;
}

function formatLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Build the merged multi-pool chart data + per-series metadata. */
export function buildAvailabilityChart(
  dto: AvailabilityDto | undefined,
  selected: string[],
  colorByPool: Record<string, string>,
): AvailabilityChart {
  const byName = new Map((dto?.pools ?? []).map((p) => [p.pool, p]));
  const present = selected.filter((p) => byName.has(p));
  const missing = selected.filter((p) => !byName.has(p));

  const series: PoolSeries[] = present.map((pool) => {
    const p = byName.get(pool)!;
    return {
      pool,
      color: colorByPool[pool] ?? POOL_PALETTE[0],
      samples: p.samples,
      avgApprox: p.avgApprox,
      availability: p.availability,
      latest: p.latest,
      bestEver: p.bestEver,
      topRegions: p.regions.slice(0, 3),
    };
  });

  // Merge every selected pool's timeseries into rows keyed by timestamp.
  const rows = new Map<string, ChartRow>();
  for (const pool of present) {
    for (const point of byName.get(pool)!.timeseries) {
      const row =
        rows.get(point.t) ?? ({ t: point.t, label: formatLabel(point.t) } as ChartRow);
      row[pool] = point.approx;
      rows.set(point.t, row);
    }
  }
  const chartData = [...rows.values()].sort((a, b) => (a.t < b.t ? -1 : 1));

  return { series, chartData, missing };
}

export function formatApprox(n: number): string {
  return `~${n.toFixed(n < 10 ? 1 : 0)}`;
}

export function formatWhen(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}
