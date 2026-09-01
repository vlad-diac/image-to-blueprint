import { BadRequestException, Injectable, Logger } from '@nestjs/common';
import { execFile } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { promisify } from 'node:util';
import {
  approxForRank,
  countRankIndex,
  periodForHour,
  poolRegions,
  PERIODS,
  RANKS,
  type AvailabilityDto,
  type PeriodStat,
  type PoolStat,
  type RegionStat,
  type Snapshot,
  type SnapshotPool,
} from './availability.types';

/** Locate scripts/gpu_availability_history relative to the repo, or via env. */
function resolveHistoryDir(): string {
  const env = process.env.GPU_HISTORY_DIR;
  if (env) return path.resolve(env);
  let dir = process.cwd();
  const candidates: string[] = [];
  for (let i = 0; i < 6; i += 1) {
    candidates.push(path.join(dir, 'scripts', 'gpu_availability_history'));
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return candidates.find((c) => fs.existsSync(c)) ?? candidates[0];
}

const execFileAsync = promisify(execFile);

@Injectable()
export class AvailabilityService {
  private readonly logger = new Logger(AvailabilityService.name);
  private readonly historyDir = resolveHistoryDir();
  private catalogCache: string[] | null = null;

  private scriptPath(): string {
    return (
      process.env.GPU_CHECK_SCRIPT ??
      path.join(path.resolve(this.historyDir, '..', '..'), 'scripts', 'check_gpu_availability.py')
    );
  }

  private python(): string {
    return process.env.GPU_CHECK_PYTHON ?? 'python3';
  }

  /** Pool ids the check script supports (cached; best-effort). */
  async getCatalog(): Promise<string[]> {
    if (this.catalogCache) return this.catalogCache;
    try {
      const { stdout } = await execFileAsync(
        this.python(),
        [this.scriptPath(), '--list-pools', '--json'],
        { timeout: 15_000 },
      );
      const parsed = JSON.parse(stdout) as { pools?: Record<string, unknown> };
      this.catalogCache = Object.keys(parsed.pools ?? {});
    } catch (err) {
      this.logger.warn(`Could not load pool catalog: ${String(err)}`);
      this.catalogCache = [];
    }
    return this.catalogCache;
  }

  /** Run the check script for the given pools (writes fresh snapshot files). */
  async runCheck(pools: string[]): Promise<void> {
    const catalog = await this.getCatalog();
    const valid = catalog.length
      ? pools.filter((p) => catalog.includes(p))
      : pools;
    if (!valid.length) {
      throw new BadRequestException(
        `No known pools in request. Valid pools: ${catalog.join(', ') || '(catalog unavailable)'}`,
      );
    }
    try {
      await execFileAsync(
        this.python(),
        [this.scriptPath(), ...valid, '--no-time-of-day'],
        { timeout: 120_000 },
      );
    } catch (err) {
      const stderr = (err as { stderr?: string }).stderr?.trim();
      this.logger.error(`check_gpu_availability failed: ${stderr ?? String(err)}`);
      throw new BadRequestException(
        stderr || `Failed to run availability check for: ${valid.join(', ')}`,
      );
    }
  }

  /** Read + parse every run-*.json snapshot, sorted oldest → newest. */
  private loadSnapshots(): Snapshot[] {
    let names: string[];
    try {
      names = fs.readdirSync(this.historyDir);
    } catch {
      this.logger.warn(`No availability history at ${this.historyDir}`);
      return [];
    }
    const snapshots: Snapshot[] = [];
    for (const name of names) {
      if (!name.startsWith('run-') || !name.endsWith('.json')) continue;
      try {
        const raw = fs.readFileSync(path.join(this.historyDir, name), 'utf8');
        const data = JSON.parse(raw) as Snapshot | Snapshot[];
        if (Array.isArray(data)) snapshots.push(...data);
        else if (data && typeof data === 'object') snapshots.push(data);
      } catch (err) {
        this.logger.warn(`Skipping unreadable snapshot ${name}: ${String(err)}`);
      }
    }
    snapshots.sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1));
    return snapshots;
  }

  private static hourOf(snap: Snapshot): number | null {
    if (typeof snap.hour === 'number') return snap.hour;
    const iso = snap.local_time ?? snap.timestamp;
    if (!iso) return null;
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? null : d.getHours();
  }

  private static bestApprox(entry: SnapshotPool): number {
    return approxForRank(Math.max(0, entry.best_rank ?? -1));
  }

  private buildPool(pool: string, snaps: Snapshot[]): PoolStat {
    const entries = snaps
      .map((s) => ({ snap: s, entry: s.pools[pool] }))
      .filter((x): x is { snap: Snapshot; entry: SnapshotPool } =>
        Boolean(x.entry),
      );

    const approxes = entries.map((e) => AvailabilityService.bestApprox(e.entry));
    const samples = entries.length;
    const avgApprox = samples ? sum(approxes) / samples : 0;
    const availability = samples
      ? entries.filter((e) => (e.entry.best_rank ?? -1) >= 1).length / samples
      : 0;

    return {
      pool,
      samples,
      avgApprox,
      availability,
      bestEver: this.bestEver(entries),
      latest: this.latest(entries),
      byTimeOfDay: this.byTimeOfDay(entries),
      regions: this.regions(entries),
      timeseries: entries.map(({ snap, entry }) => {
        const approx = AvailabilityService.bestApprox(entry);
        return { t: snap.timestamp, approx, rank: RANKS[countRankIndex(approx)].name };
      }),
    };
  }

  private bestEver(
    entries: { snap: Snapshot; entry: SnapshotPool }[],
  ): PoolStat['bestEver'] {
    let best: PoolStat['bestEver'] = null;
    for (const { snap, entry } of entries) {
      const rank = entry.best_rank ?? -1;
      if (!best || rank > best.rank) {
        best = {
          rank,
          stock: entry.best_stock,
          dc: entry.best_dc,
          gpu: entry.best_gpu,
          seenAt: snap.timestamp,
        };
      }
    }
    return best;
  }

  private latest(
    entries: { snap: Snapshot; entry: SnapshotPool }[],
  ): PoolStat['latest'] {
    const last = entries[entries.length - 1];
    if (!last) return null;
    return {
      approx: AvailabilityService.bestApprox(last.entry),
      stock: last.entry.best_stock,
      dc: last.entry.best_dc,
      at: last.snap.timestamp,
    };
  }

  private byTimeOfDay(
    entries: { snap: Snapshot; entry: SnapshotPool }[],
  ): PeriodStat[] {
    const buckets = new Map<string, number[]>(PERIODS.map((p) => [p.name, []]));
    for (const { snap, entry } of entries) {
      const hour = AvailabilityService.hourOf(snap);
      const period = hour == null ? null : periodForHour(hour);
      if (period) buckets.get(period)!.push(AvailabilityService.bestApprox(entry));
    }
    return PERIODS.map((p) => {
      const vals = buckets.get(p.name)!;
      return {
        period: p.name,
        avg: vals.length ? sum(vals) / vals.length : null,
        samples: vals.length,
      };
    });
  }

  private regions(
    entries: { snap: Snapshot; entry: SnapshotPool }[],
  ): RegionStat[] {
    const acc = new Map<string, { counts: number[]; location: string }>();
    for (const { entry } of entries) {
      for (const reg of poolRegions(entry)) {
        if (!reg.dc) continue;
        const slot = acc.get(reg.dc) ?? { counts: [], location: '' };
        slot.counts.push(approxForRank(Math.max(0, reg.rank ?? -1)));
        if (reg.location) slot.location = reg.location;
        acc.set(reg.dc, slot);
      }
    }
    const total = entries.length || 1;
    const regions: RegionStat[] = [...acc.entries()].map(([dc, slot]) => ({
      dc,
      location: slot.location,
      avgApprox: sum(slot.counts) / slot.counts.length,
      samples: slot.counts.length,
      share: slot.counts.length / total,
    }));
    regions.sort(
      (a, b) => b.avgApprox - a.avgApprox || b.samples - a.samples || a.dc.localeCompare(b.dc),
    );
    return regions;
  }

  async getAvailability(): Promise<AvailabilityDto> {
    const snaps = this.loadSnapshots();
    const poolNames = [
      ...new Set(snaps.flatMap((s) => Object.keys(s.pools ?? {}))),
    ].sort();
    const catalog = await this.getCatalog();

    const pools = poolNames.map((p) => this.buildPool(p, snaps));

    // overall: rank distribution + mean approx across every pool-run sample
    const dist = new Map<string, number>(RANKS.map((r) => [r.name, 0]));
    const allApprox: number[] = [];
    for (const s of snaps) {
      for (const entry of Object.values(s.pools ?? {})) {
        const idx = Math.max(0, entry.best_rank ?? -1);
        dist.set(RANKS[idx].name, (dist.get(RANKS[idx].name) ?? 0) + 1);
        allApprox.push(approxForRank(idx));
      }
    }

    const periodSamples: Record<string, number> = Object.fromEntries(
      PERIODS.map((p) => [p.name, 0]),
    );
    for (const s of snaps) {
      const hour = AvailabilityService.hourOf(s);
      const period = hour == null ? null : periodForHour(hour);
      if (period) periodSamples[period] += 1;
    }

    return {
      runs: snaps.length,
      firstRun: snaps[0]?.timestamp ?? null,
      lastRun: snaps[snaps.length - 1]?.timestamp ?? null,
      rankRanges: RANKS,
      periods: PERIODS.map((p) => p.name),
      periodSamples,
      catalog: catalog.length ? catalog : poolNames,
      overall: {
        avgApprox: allApprox.length ? sum(allApprox) / allApprox.length : 0,
        rankDistribution: RANKS.map((r) => ({
          rank: r.name,
          count: dist.get(r.name) ?? 0,
        })),
      },
      pools,
    };
  }
}

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}
