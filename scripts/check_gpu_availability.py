#!/usr/bin/env python3
"""
Check RunPod GPU availability per datacenter/region, grouped by GPU pool.

Uses `runpodctl datacenter list`, which returns every datacenter with its
per-GPU `stockStatus` (High/Medium/Low/None) in one call — the same source
worker/scripts/deploy.sh already relies on.

GPU pools are RunPod's own groupings (see the GPU types docs). Each pool below
maps to the exact `gpuId`s it contains, so matching is unambiguous
(e.g. `AMPERE_24`'s "L4" never matches "L40"/"L40S").

Usage:
  # Default pools (see DEFAULT_POOLS below)
  python scripts/check_gpu_availability.py

  # Specific pools, or every pool
  python scripts/check_gpu_availability.py ADA_80_PRO AMPERE_80
  python scripts/check_gpu_availability.py all

  # Only a region, only stock >= Medium, machine-readable output
  python scripts/check_gpu_availability.py ADA_80_PRO --region Europe
  python scripts/check_gpu_availability.py ADA_80_PRO --min-stock Medium
  python scripts/check_gpu_availability.py ADA_80_PRO --json

  # List the pool → gpuId mapping and exit
  python scripts/check_gpu_availability.py --list-pools

  # For polling loops / CI: exit non-zero if nothing meets the threshold
  python scripts/check_gpu_availability.py ADA_80_PRO --min-stock Medium --fail-empty

Every run saves a timestamped snapshot as its own JSON file in a history folder
(default: scripts/gpu_availability_history/, committed to git). On each run the
script loads *all* snapshot files in that folder, merges them, and:
  • prints an "availability by time of day" summary (local-time buckets), so
    repeated polling reveals when each pool tends to have stock, and
  • (re)writes the merged "official" list — official.json — the consolidated
    view you compare individual runs against.

  python scripts/check_gpu_availability.py --history-dir /tmp/gpu-history
  python scripts/check_gpu_availability.py --no-save        # don't record this run
  python scripts/check_gpu_availability.py --no-time-of-day # skip the summary

Requires: runpodctl installed + authenticated (https://docs.runpod.io/runpodctl/install).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

# RunPod Flash GPU pools (GpuGroup) → the exact gpuId strings each contains,
# from the Flash GPU types docs. Kept strict to that page: e.g. ADA_80_PRO is
# H100 80GB HBM3 only (not H100 PCIe/NVL), and HOPPER_141 is H200 only.
POOLS: dict[str, list[str]] = {
    "AMPERE_16": [
        "NVIDIA RTX A4000",
        "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA RTX 2000 Ada Generation",
    ],
    "AMPERE_24": [
        "NVIDIA RTX A4500",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
    ],
    "ADA_24": [
        "NVIDIA L4",
        "NVIDIA GeForce RTX 4090",
    ],
    "ADA_32_PRO": [
        "NVIDIA GeForce RTX 5090",
    ],
    "AMPERE_48": [
        "NVIDIA A40",
        "NVIDIA RTX A6000",
    ],
    "ADA_48_PRO": [
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA RTX 6000 Ada Generation",
    ],
    "AMPERE_80": [
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    ],
    "ADA_80_PRO": [
        "NVIDIA H100 80GB HBM3",
    ],
    "BLACKWELL_96": [
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ],
    "HOPPER_141": [
        "NVIDIA H200",
    ],
    "BLACKWELL_180": [
        "NVIDIA B200",
    ],
}

# Pools checked when none are passed on the command line.
DEFAULT_POOLS: list[str] = ["ADA_80_PRO", "AMPERE_80", "ADA_48_PRO", "ADA_24"]

# High > Medium > Low > None; unknown/empty sorts last.
STOCK_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}
STOCK_ORDER = ["None", "Low", "Medium", "High"]  # for --min-stock choices

# Snapshots are bucketed by local hour-of-day into these named periods so
# repeated polling reveals when each pool tends to have stock.
PERIODS: list[tuple[str, range]] = [
    ("Night", range(0, 6)),      # 00:00–05:59
    ("Morning", range(6, 12)),   # 06:00–11:59
    ("Afternoon", range(12, 18)),  # 12:00–17:59
    ("Evening", range(18, 24)),  # 18:00–23:59
]

# Each run writes one snapshot file (run-<UTC>.json) into this folder, and the
# merged/consolidated result is (re)written to official.json alongside them.
# The folder is committed to git so runs can be compared over time.
DEFAULT_HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gpu_availability_history",
)
OFFICIAL_FILENAME = "official.json"


def _stock_rank(status: str | None) -> int:
    return STOCK_RANK.get((status or "").strip().lower(), -1)


# ── color (respects NO_COLOR and non-tty) ─────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_RESET = "\033[0m"
_BOLD, _DIM, _CYAN = "1", "2", "0;36"
_STOCK_COLOR = {"high": "0;32", "medium": "1;33", "low": "0;31", "none": "2"}


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}{_RESET}" if _USE_COLOR else text


def _stock_str(status: str | None) -> str:
    label = status or "unknown"
    code = _STOCK_COLOR.get((status or "").strip().lower())
    return _c(code, label) if code else label


def _which(name: str) -> bool:
    return any(
        os.access(os.path.join(p, name), os.X_OK)
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    )


def resolve_pools(names: list[str]) -> list[str]:
    """Validate/expand pool names ('all' → every pool). Raises SystemExit on unknown."""
    if not names:
        return list(DEFAULT_POOLS)
    if any(n.lower() == "all" for n in names):
        return list(POOLS)
    resolved, unknown = [], []
    for n in names:
        key = n.upper()
        if key in POOLS:
            if key not in resolved:
                resolved.append(key)
        else:
            unknown.append(n)
    if unknown:
        raise SystemExit(
            f"Unknown pool(s): {', '.join(unknown)}\n"
            f"Valid pools: {', '.join(POOLS)} (or 'all').",
        )
    return resolved


def _run_datacenter_list(cmd: list[str]) -> list[dict[str, Any]]:
    """Run one runpodctl variant; return parsed datacenters or raise ValueError."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(str(exc)) from exc
    if out.returncode != 0:
        raise ValueError((out.stderr or out.stdout).strip())
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON: {exc}") from exc
    if isinstance(data, dict):  # some versions wrap in {"dataCenters": [...]}
        data = data.get("dataCenters") or data.get("data") or []
    return data or []


def fetch_datacenters() -> list[dict[str, Any]]:
    """Return `runpodctl datacenter list` JSON (retries with `-o json`)."""
    if not _which("runpodctl"):
        raise SystemExit(
            "runpodctl not found. Install: https://docs.runpod.io/runpodctl/install",
        )
    last_err = ""
    for cmd in (["runpodctl", "datacenter", "list"],
                ["runpodctl", "datacenter", "list", "-o", "json"]):
        try:
            return _run_datacenter_list(cmd)
        except ValueError as exc:
            last_err = str(exc)
    raise SystemExit(f"runpodctl datacenter list failed: {last_err}")


def _in_region(dc: dict[str, Any], region: str | None) -> bool:
    if not region:
        return True
    needle = region.lower()
    return needle in (dc.get("location") or "").lower() \
        or needle in (dc.get("id") or "").lower()


def _collect_dc(
    report: dict[str, dict[str, list[dict[str, Any]]]],
    gpu_to_pool: dict[str, str],
    dc: dict[str, Any],
    min_rank: int,
) -> None:
    """Append this datacenter's matching GPU rows into `report[pool][gpu_id]`."""
    for gpu in dc.get("gpuAvailability") or []:
        gpu_id = gpu.get("gpuId") or ""
        pool = gpu_to_pool.get(gpu_id)
        if pool is None:
            continue
        rank = _stock_rank(gpu.get("stockStatus"))
        if rank < min_rank:
            continue
        report[pool].setdefault(gpu_id, []).append({
            "dc": dc.get("id") or dc.get("name") or "?",
            "location": dc.get("location") or "",
            "display": gpu.get("displayName") or gpu_id,
            "stock": gpu.get("stockStatus"),
            "rank": rank,
        })


def build_report(
    datacenters: list[dict[str, Any]],
    pools: list[str],
    region: str | None,
    min_rank: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """pool → gpuId → [{dc, location, display, stock, rank}], ranked desc."""
    gpu_to_pool = {gpu_id: pool for pool in pools for gpu_id in POOLS[pool]}
    report: dict[str, dict[str, list[dict[str, Any]]]] = {p: {} for p in pools}
    for dc in datacenters:
        if _in_region(dc, region):
            _collect_dc(report, gpu_to_pool, dc, min_rank)
    for gpus in report.values():
        for rows in gpus.values():
            rows.sort(key=lambda r: (-r["rank"], r["dc"]))
    return report


def _print_pool(pool: str, gpus: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Print one pool's datacenter rows; return its best row (or None if empty)."""
    print(f"\n{_c(_BOLD, pool)}  {_c(_DIM, '(' + ', '.join(POOLS[pool]) + ')')}")
    if not gpus:
        print(f"  {_c(_DIM, 'no availability')}")
        return None
    best = None
    for gpu_id in sorted(gpus):
        rows = gpus[gpu_id]
        print(f"  {gpu_id}  {_c(_DIM, f'({len(rows)} datacenter(s))')}")
        for r in rows:
            loc = f" — {r['location']}" if r["location"] else ""
            print(f"    {_c(_CYAN, r['dc']):<26}{loc:<22} stock: {_stock_str(r['stock'])}")
        if best is None or rows[0]["rank"] > best["rank"]:
            best = {**rows[0], "gpu": gpu_id}
    return best


def print_report(report: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    best_lines = []
    for pool, gpus in report.items():
        best = _print_pool(pool, gpus)
        if best:
            loc = f" ({best['location']})" if best["location"] else ""
            best_lines.append(
                f"  {pool:<12} → {_c(_CYAN, best['dc'])}{loc}  "
                f"{best['gpu']}  {_stock_str(best['stock'])}",
            )

    if best_lines:
        print(f"\n{_c(_BOLD, 'Best pick per pool')}")
        print("\n".join(best_lines))


def print_pools() -> None:
    print(_c(_BOLD, "RunPod GPU pools"))
    for pool, gpu_ids in POOLS.items():
        print(f"\n{_c(_CYAN, pool)}")
        for gpu_id in gpu_ids:
            print(f"  {gpu_id}")


# ── history / time-of-day ─────────────────────────────────────────────────────
def summarize_report(
    report: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Reduce a full report to per-pool best pick + datacenter count, for storage."""
    summary: dict[str, dict[str, Any]] = {}
    for pool, gpus in report.items():
        best: dict[str, Any] | None = None
        dc_count = 0
        for gpu_id, rows in gpus.items():
            dc_count += len(rows)
            if rows and (best is None or rows[0]["rank"] > best["rank"]):
                best = {**rows[0], "gpu": gpu_id}
        summary[pool] = {
            "best_rank": best["rank"] if best else -1,
            "best_stock": best["stock"] if best else None,
            "best_dc": best["dc"] if best else None,
            "best_gpu": best["gpu"] if best else None,
            "dc_count": dc_count,
        }
    return summary


def make_snapshot(
    summary: dict[str, dict[str, Any]], region: str | None,
) -> dict[str, Any]:
    """Build the record appended to history for this run (UTC + local time)."""
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone()
    return {
        "timestamp": now_utc.isoformat(timespec="seconds"),
        "local_time": now_local.isoformat(timespec="seconds"),
        "hour": now_local.hour,
        "region": region,
        "pools": summary,
    }


def _file_stamp(iso_utc: str) -> str:
    """Filesystem-safe, sortable UTC stamp for a snapshot filename."""
    try:
        return datetime.fromisoformat(iso_utc).astimezone(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ",
        )
    except (ValueError, TypeError):
        return "unknown"


def save_snapshot(history_dir: str, record: dict[str, Any]) -> str:
    """Write one run's snapshot as run-<UTC>.json (unique). Returns the path."""
    os.makedirs(history_dir, exist_ok=True)
    stamp = _file_stamp(record.get("timestamp", ""))
    path = os.path.join(history_dir, f"run-{stamp}.json")
    n = 1
    while os.path.exists(path):  # multiple runs in the same second
        path = os.path.join(history_dir, f"run-{stamp}-{n}.json")
        n += 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return path


def load_history(history_dir: str) -> list[dict[str, Any]]:
    """Load + merge every run-*.json snapshot in the folder (skips official.json).

    Corrupt/unreadable files are skipped rather than failing the run.
    """
    if not os.path.isdir(history_dir):
        return []
    records: list[dict[str, Any]] = []
    for name in sorted(os.listdir(history_dir)):
        if not name.startswith("run-") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(history_dir, name), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            records.append(data)
        elif isinstance(data, list):  # tolerate a file holding several records
            records.extend(r for r in data if isinstance(r, dict))
    return records


def build_official(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge all runs into the consolidated 'official' list.

    Covers every pool seen across all snapshots (not just the current run's) and
    combines the availability-by-time-of-day view with each pool's best-ever pick.
    """
    pools = sorted({p for r in records for p in (r.get("pools") or {})})
    best_ever: dict[str, Any] = {}
    for r in records:
        for pool, entry in (r.get("pools") or {}).items():
            rank = int(entry.get("best_rank", -1))
            cur = best_ever.get(pool)
            if cur is None or rank > cur["best_rank"]:
                best_ever[pool] = {
                    "best_rank": rank,
                    "best_stock": entry.get("best_stock"),
                    "best_dc": entry.get("best_dc"),
                    "best_gpu": entry.get("best_gpu"),
                    "seen_at": r.get("timestamp"),
                }
    stamps = sorted(r["timestamp"] for r in records if r.get("timestamp"))
    return {
        "runs": len(records),
        "first_run": stamps[0] if stamps else None,
        "last_run": stamps[-1] if stamps else None,
        "pools": pools,
        "time_of_day": aggregate_by_time_of_day(records, pools),
        "best_ever": best_ever,
    }


def write_official(history_dir: str, records: list[dict[str, Any]]) -> str:
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, OFFICIAL_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_official(records), fh, indent=2)
        fh.write("\n")
    return path


def _record_hour(record: dict[str, Any]) -> int | None:
    """Local hour-of-day for a record (stored, or derived from its timestamps)."""
    hour = record.get("hour")
    if isinstance(hour, int):
        return hour
    for key in ("local_time", "timestamp"):
        ts = record.get(key)
        if not ts:
            continue
        try:
            return datetime.fromisoformat(ts).astimezone().hour
        except ValueError:
            continue
    return None


def aggregate_by_time_of_day(
    records: list[dict[str, Any]], pools: list[str],
) -> dict[str, Any]:
    """pool → period → {'avg': float|None, 'samples': int} over all snapshots."""
    period_of = {h: name for name, hours in PERIODS for h in hours}
    # pool → period → list of best_rank values (clamped to >= 0)
    acc: dict[str, dict[str, list[int]]] = {
        p: {name: [] for name, _ in PERIODS} for p in pools
    }
    period_samples = {name: 0 for name, _ in PERIODS}
    for record in records:
        hour = _record_hour(record)
        period = period_of.get(hour) if hour is not None else None
        if period is None:
            continue
        period_samples[period] += 1
        rec_pools = record.get("pools") or {}
        for pool in pools:
            entry = rec_pools.get(pool)
            if entry is None:
                continue
            acc[pool][period].append(max(0, int(entry.get("best_rank", -1))))
    result: dict[str, Any] = {"period_samples": period_samples, "pools": {}}
    for pool in pools:
        result["pools"][pool] = {
            name: {
                "avg": (sum(vals) / len(vals)) if vals else None,
                "samples": len(vals),
            }
            for name, vals in acc[pool].items()
        }
    return result


def print_time_of_day(agg: dict[str, Any], pools: list[str], total: int) -> None:
    print(f"\n{_c(_BOLD, 'Availability by time of day')}  "
          f"{_c(_DIM, f'(local time, {total} snapshot(s))')}")
    if total == 0:
        print(f"  {_c(_DIM, 'no history yet — run again over time to build it up')}")
        return

    names = [name for name, _ in PERIODS]
    samples = agg["period_samples"]
    headers = ["pool", *(f"{name} (n={samples[name]})" for name in names)]

    def cell_text(pool: str, name: str) -> str:
        stat = agg["pools"][pool][name]
        if stat["avg"] is None:
            return "—"
        label = STOCK_ORDER[max(0, min(3, round(stat["avg"])))]
        return f"{label} {stat['avg']:.1f}"

    rows = [[pool, *(cell_text(pool, name) for name in names)] for pool in pools]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    print("  " + "  ".join(_c(_BOLD, h.ljust(widths[i])) for i, h in enumerate(headers)))
    for pool, row in zip(pools, rows):
        cells = [_c(_CYAN, row[0].ljust(widths[0]))]
        for i, name in enumerate(names, start=1):
            text = row[i].ljust(widths[i])
            stat = agg["pools"][pool][name]
            code = None
            if stat["avg"] is not None:
                label = STOCK_ORDER[max(0, min(3, round(stat["avg"])))]
                code = _STOCK_COLOR.get(label.lower())
            cells.append(_c(code, text) if code else text)
        print("  " + "  ".join(cells))
    print(f"  {_c(_DIM, 'avg stock per period: None 0 · Low 1 · Medium 2 · High 3')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check RunPod GPU availability per datacenter/region, by GPU pool.",
    )
    parser.add_argument(
        "pools", nargs="*",
        help=f"Pool IDs to check, or 'all' (default: {', '.join(DEFAULT_POOLS)}). "
             f"Choices: {', '.join(POOLS)}.",
    )
    parser.add_argument(
        "--region",
        help="Only datacenters whose location or id contains this (e.g. Europe, US, EU-RO-1).",
    )
    parser.add_argument(
        "--min-stock", choices=STOCK_ORDER, default="None",
        help="Only show availability at or above this level (default: None = show all).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--list-pools", action="store_true", help="Print the pool → gpuId map and exit.")
    parser.add_argument(
        "--fail-empty", action="store_true",
        help="Exit non-zero if no datacenter meets the threshold for any selected pool.",
    )
    parser.add_argument(
        "--history-dir", default=DEFAULT_HISTORY_DIR,
        help=f"Folder of per-run snapshot files + official.json (default: {DEFAULT_HISTORY_DIR}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't write this run's snapshot or update the official list.",
    )
    parser.add_argument(
        "--no-time-of-day", action="store_true",
        help="Don't print the availability-by-time-of-day summary.",
    )
    args = parser.parse_args(argv)

    if args.list_pools:
        print_pools()
        return 0

    pools = resolve_pools(args.pools)
    min_rank = _stock_rank(args.min_stock)

    datacenters = fetch_datacenters()
    report = build_report(datacenters, pools, args.region, min_rank)

    # Record an unfiltered-by-stock snapshot so history stays comparable across
    # runs regardless of the display filter (region scoping is preserved + noted).
    snapshot = make_snapshot(
        summarize_report(build_report(datacenters, pools, args.region, -1)),
        args.region,
    )
    saved_path = official_path = None
    if not args.no_save:
        try:
            saved_path = save_snapshot(args.history_dir, snapshot)
        except OSError as exc:
            print(f"warning: could not write snapshot file: {exc}", file=sys.stderr)

    # Load + merge every run in the folder (includes the one just saved).
    history = load_history(args.history_dir)
    if args.no_save:  # not persisted, but still count this run in the merge
        history.append(snapshot)
    else:  # keep the merged official list current
        try:
            official_path = write_official(args.history_dir, history)
        except OSError as exc:
            print(f"warning: could not write official list: {exc}", file=sys.stderr)

    agg = aggregate_by_time_of_day(history, pools)

    if args.json:
        print(json.dumps({
            "timestamp": snapshot["timestamp"],
            "report": report,
            "time_of_day": agg,
            "official": build_official(history),
        }, indent=2))
    else:
        print_report(report)
        if not args.no_time_of_day:
            print_time_of_day(agg, pools, len(history))
        if saved_path or official_path:
            hint = f"  {_c(_DIM, f'saved {os.path.basename(saved_path)}')}" if saved_path else ""
            off = f"  {_c(_DIM, f'· official list: {OFFICIAL_FILENAME} ({len(history)} runs)')}" if official_path else ""
            print(f"\n{_c(_DIM, 'history:')}{hint}{off}")

    if args.fail_empty and not any(report.values()):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
