/**
 * Types + aggregation helpers for GPU-availability history.
 *
 * The data source is the folder of per-run snapshots written by
 * `scripts/check_gpu_availability.py` (run-<UTC>.json). RunPod exposes only a
 * coarse category (None/Low/Medium/High), never an exact GPU count, so each rank
 * maps to an approximate count range — this MUST stay in sync with the `RANKS`
 * constant in that script.
 */

export interface RankSpec {
  name: string;
  lo: number;
  hi: number | null;
  approx: number;
}

export const RANKS: RankSpec[] = [
  { name: 'None', lo: 0, hi: 0, approx: 0 },
  { name: 'Low', lo: 1, hi: 4, approx: 2 },
  { name: 'Medium', lo: 5, hi: 9, approx: 7 },
  { name: 'High', lo: 10, hi: null, approx: 14 },
];

export const PERIODS: { name: string; start: number; end: number }[] = [
  { name: 'Night', start: 0, end: 6 },
  { name: 'Morning', start: 6, end: 12 },
  { name: 'Afternoon', start: 12, end: 18 },
  { name: 'Evening', start: 18, end: 24 },
];

/** Representative GPU count for a stored rank index (0 if unknown). */
export function approxForRank(index: number): number {
  return index >= 0 && index < RANKS.length ? RANKS[index].approx : 0;
}

/** Rank index whose range contains this (approximate) GPU count. */
export function countRankIndex(count: number): number {
  const c = Math.max(0, Math.round(count));
  for (let i = 0; i < RANKS.length; i += 1) {
    const { lo, hi } = RANKS[i];
    if (c >= lo && c <= (hi ?? c)) return i;
  }
  return RANKS.length - 1;
}

export function periodForHour(hour: number): string | null {
  const p = PERIODS.find((x) => hour >= x.start && hour < x.end);
  return p ? p.name : null;
}

// ── raw snapshot shape (as written by the Python script) ──────────────────────
export interface SnapshotRegion {
  dc: string;
  location: string;
  rank: number;
  stock: string | null;
}

export interface SnapshotPool {
  best_rank: number;
  best_stock: string | null;
  best_dc: string | null;
  best_gpu: string | null;
  dc_count: number;
  regions?: SnapshotRegion[];
}

export interface Snapshot {
  timestamp: string;
  local_time?: string;
  hour?: number;
  region?: string | null;
  pools: Record<string, SnapshotPool>;
}

/** Per-region rows for a pool entry (falls back to best_dc for old snapshots). */
export function poolRegions(entry: SnapshotPool): SnapshotRegion[] {
  if (entry.regions && entry.regions.length) return entry.regions;
  if (!entry.best_dc) return [];
  return [
    {
      dc: entry.best_dc,
      location: '',
      rank: entry.best_rank ?? -1,
      stock: entry.best_stock,
    },
  ];
}

// ── API response shape ────────────────────────────────────────────────────────
export interface RegionStat {
  dc: string;
  location: string;
  avgApprox: number;
  samples: number;
  share: number; // fraction of the pool's runs this region appeared in
}

export interface PeriodStat {
  period: string;
  avg: number | null; // mean approx GPUs; null = no samples in this period
  samples: number;
}

export interface PoolStat {
  pool: string;
  samples: number;
  avgApprox: number;
  availability: number; // fraction of runs with any stock (rank >= Low)
  bestEver: {
    rank: number;
    stock: string | null;
    dc: string | null;
    gpu: string | null;
    seenAt: string | null;
  } | null;
  latest: {
    approx: number;
    stock: string | null;
    dc: string | null;
    at: string | null;
  } | null;
  byTimeOfDay: PeriodStat[];
  regions: RegionStat[];
  timeseries: { t: string; approx: number; rank: string }[];
}

export interface AvailabilityDto {
  runs: number;
  firstRun: string | null;
  lastRun: string | null;
  rankRanges: RankSpec[];
  periods: string[];
  periodSamples: Record<string, number>;
  /** Every pool the check script knows about (for the "add pool" control). */
  catalog: string[];
  overall: {
    avgApprox: number;
    rankDistribution: { rank: string; count: number }[];
  };
  pools: PoolStat[];
}
