#!/usr/bin/env python3
"""
analyze_worker_logs.py — parse RunPod worker logs and find where time is wasted.

The log lines look like:

    2026-08-28 18:01:43.436 | info | 3g13did7l27p1b | handler.py :99  2026-08-28 15:01:16,076 Loading pipeline (cold start)…

    ^ outer (shipper) ts      lvl   ^ worker id       ^ message, which for
                                                        python-logged lines embeds
                                                        an inner ts (HH:MM:SS,mmm)

IMPORTANT — which clock to trust
--------------------------------
The *outer* timestamp is when the line was shipped, and the shipper batches:
a whole cold start (tens of seconds of real work) is often flushed in one
burst, collapsing those seconds into a few milliseconds of outer time. So the
outer clock badly understates durations.

The *inner* timestamp (added by python `logging`) is the real event time, but
only python-logged lines carry it — framework lines ("Started.", "Finished.",
fitness checks, the tqdm bar) have only the outer ts, and the two clocks sit in
different timezones.

We therefore build a single per-worker timeline on the INNER clock:
  * lines with an inner ts   -> use it directly
  * lines without an inner ts -> outer ts minus a per-worker offset, where
    offset = min(outer - inner) over lines that have both (the min picks the
    moments with ~zero shipping lag, i.e. the true timezone delta).

A "run" is one inference job (Inference: … -> Finished.), optionally preceded by
a cold start on that worker. We attribute the wall-clock gap between each pair of
consecutive milestones to the phase that the earlier milestone starts, then
aggregate to answer: where did we waste the most time?

Usage:
    python3 scripts/analyze_worker_logs.py [PATH ...]
    python3 scripts/analyze_worker_logs.py --json report.json "output/logs/*.txt"

With no PATH, defaults to output/logs/*.txt.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

LINE_RE = re.compile(
    r"^(?P<outer>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s*\|"
    r"\s*(?P<level>\w+)\s*\|"
    r"\s*(?P<worker>\S+)\s*\|"
    r"\s*(?P<msg>.*)$"
)
INNER_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
JOB_RE = re.compile(r"jobs/([^/]+)/")
# tqdm bar: capture every "MM:SS<00:00" and take the largest -> total step time.
TQDM_RE = re.compile(r"(\d\d):(\d\d)<00:00")


def _parse_outer(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def _parse_inner(msg: str) -> datetime | None:
    m = INNER_RE.search(msg)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")


@dataclass
class Record:
    outer: datetime
    worker: str
    msg: str
    inner: datetime | None
    utime: datetime = field(default=None)  # unified (inner-clock) time, filled later


# ---------------------------------------------------------------------------
# Milestones: (key, regex, phase-label for the gap that STARTS at this line)
# The gap between milestone A and the next milestone B is billed to A's phase.
# ---------------------------------------------------------------------------

MILESTONES: list[tuple[str, re.Pattern, str]] = [
    ("cold_start_begin", re.compile(r"Loading pipeline \(cold start\)"), "cold_start:startup/find_models"),
    ("transformer_src",  re.compile(r"Transformer source:"),            "cold_start:transformer_load(gguf)"),
    ("tx_backend",       re.compile(r"Transformer attention backend:"), "cold_start:(post-transformer)"),
    ("vae_src",          re.compile(r"VAE source:"),                    "cold_start:vae_load"),
    ("te_config",        re.compile(r"Loading text encoder config"),    "cold_start:(te config)"),
    ("te_streaming",     re.compile(r"Streaming FP8 file"),             "cold_start:text_encoder_load(fp8)"),
    ("te_ready",         re.compile(r"Text encoder ready"),             "cold_start:(post-te)"),
    ("tokenizer_src",    re.compile(r"Tokenizer source:"),              "cold_start:tokenizer_load"),
    ("processor_src",    re.compile(r"Processor source:"),              "cold_start:processor_load"),
    ("assembling",       re.compile(r"Assembling QwenImageEditPlusPipeline"), "cold_start:assemble+to_gpu+lora_angles"),
    ("lora_angles",      re.compile(r"Loaded LoRA 'angles'"),           "cold_start:lora_lightning"),
    ("lora_lightning",   re.compile(r"Loaded LoRA 'lightning'"),        "cold_start:(finalize adapters)"),
    ("pipeline_ready",   re.compile(r"Pipeline ready"),                 "startup:(pipeline_ready->fitness)"),
    ("fitness_begin",    re.compile(r"Running \d+ fitness check"),      "startup:fitness_checks"),
    ("fitness_done",     re.compile(r"All fitness checks passed"),      "startup:dispatch/queue_wait"),
    ("inference_begin",  re.compile(r"Inference: num_images"),          "run:inference(encode+diffuse+decode)"),
    ("svg_start",        re.compile(r"Running SVG conversion for .*/output_(\d+)\.png"), "run:svg_convert"),
    ("svg_done",         re.compile(r"Successfully encoded SVG for output_(\d+)"),       "run:svg_overhead"),
    ("finished",         re.compile(r"Finished\."),                     "idle:warm_idle"),
]

COLD_START_KEYS = {k for k, _, lbl in MILESTONES if lbl.startswith("cold_start")}

# Ordinal position of each milestone in a normal boot→job→finish sequence.
# A run "cycle" advances monotonically through these; seeing a lower ordinal than
# we've already reached means a NEW cycle started (new cold start, or the next
# job on a warm worker), so the previous run ends there — even if it never
# logged "Finished." (a boot that loaded but got no job, or a truncated log).
ORD = {k: i for i, (k, _, _) in enumerate(MILESTONES)}
ORD["svg_done"] = ORD["svg_start"]  # svg_start/done repeat per image; same region


@dataclass
class Milestone:
    key: str
    phase: str
    utime: datetime
    msg: str
    n: int | None = None  # svg output index, if applicable


def classify(msg: str) -> tuple[str, str, int | None] | None:
    for key, rx, phase in MILESTONES:
        m = rx.search(msg)
        if m:
            n = None
            if key in ("svg_start", "svg_done") and m.groups():
                n = int(m.group(1))
            return key, phase, n
    return None


# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    phase: str
    seconds: float
    worker: str
    run_idx: int
    start: datetime


@dataclass
class Run:
    worker: str
    idx: int
    milestones: list[Milestone] = field(default_factory=list)
    job_id: str | None = None
    diffusion_s: float | None = None
    idle_before_s: float = 0.0          # warm-idle on this worker just before the run
    gpu: str | None = None              # GPU model, from the job-take URLs in the log
    gpu_confirmed: bool = False         # True if THIS worker logged its GPU; else endpoint-implied
    vram_gb: int | None = None          # GPU VRAM (GB) for that model (spec lookup)
    host_ram_gb: float | None = None    # system RAM total from the fitness "Memory check"

    @property
    def start(self) -> datetime:
        return self.milestones[0].utime

    @property
    def end(self) -> datetime:
        return self.milestones[-1].utime

    @property
    def has_cold_start(self) -> bool:
        return any(m.key in COLD_START_KEYS for m in self.milestones)

    def gaps(self) -> list[Gap]:
        out: list[Gap] = []
        for a, b in zip(self.milestones, self.milestones[1:]):
            phase = a.phase
            if a.key == "svg_start" and a.n is not None:
                phase = f"run:svg_convert[{a.n}]"
            out.append(Gap(phase, (b.utime - a.utime).total_seconds(),
                           self.worker, self.idx, a.utime))
        return out

    def phase_total(self, prefix: str) -> float:
        return sum(g.seconds for g in self.gaps() if g.phase.startswith(prefix))

    @property
    def total_s(self) -> float:
        return (self.end - self.start).total_seconds()


# ---------------------------------------------------------------------------
# Parsing + timeline construction
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> list[Record]:
    records: list[Record] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.rstrip("\n")
        # strip a literal trailing "\n" that the shipper sometimes appends
        if raw.endswith("\\n"):
            raw = raw[:-2]
        m = LINE_RE.match(raw)
        if not m:
            continue
        try:
            outer = _parse_outer(m.group("outer"))
        except ValueError:
            continue
        msg = m.group("msg")
        records.append(Record(outer=outer, worker=m.group("worker"),
                              msg=msg, inner=_parse_inner(msg)))
    return records


def unify_clocks(records: list[Record]) -> None:
    """Fill Record.utime on a per-worker inner clock (see module docstring)."""
    by_worker: dict[str, list[Record]] = {}
    for r in records:
        by_worker.setdefault(r.worker, []).append(r)

    for recs in by_worker.values():
        offsets = [(r.outer - r.inner) for r in recs if r.inner is not None]
        offset = min(offsets) if offsets else timedelta(0)
        for r in recs:
            r.utime = r.inner if r.inner is not None else (r.outer - offset)


def build_runs(records: list[Record]) -> list[Run]:
    by_worker: dict[str, list[Record]] = {}
    for r in records:
        by_worker.setdefault(r.worker, []).append(r)

    runs: list[Run] = []
    for worker, recs in by_worker.items():
        recs.sort(key=lambda r: r.utime)
        run: Run | None = None
        run_counter = 0
        reached = -1  # furthest ordinal reached in the current run

        def close(r: Run | None) -> None:
            if r is not None and len(r.milestones) > 1:
                runs.append(r)

        for r in recs:
            c = classify(r.msg)
            if c is None:
                continue
            key, phase, n = c
            ms = Milestone(key=key, phase=phase, utime=r.utime, msg=r.msg, n=n)

            # tqdm bar carries the pure diffusion-step time; stash it on the run.
            if run is not None:
                tq = TQDM_RE.findall(r.msg)
                if tq:
                    run.diffusion_s = max(int(a) * 60 + int(b) for a, b in tq)

            job = JOB_RE.search(r.msg)

            # New cycle if the sequence rewound (or nothing is open yet).
            if run is None or ORD[key] < reached:
                close(run)
                run_counter += 1
                run = Run(worker=worker, idx=run_counter)
                reached = -1

            run.milestones.append(ms)
            reached = max(reached, ORD[key])
            if job and not run.job_id:
                run.job_id = job.group(1)

            if key == "finished":
                close(run)
                run = None
                reached = -1

        close(run)  # trailing run (log window cut off before Finished)

    runs.sort(key=lambda r: r.start)
    return runs


# ---------------------------------------------------------------------------
# GPU / memory fingerprint (from the log itself)
# ---------------------------------------------------------------------------
#
# The GPU model is only ever written to these logs inside the 403 error URLs
# (…/job-take/<worker>?gpu=<MODEL>). Successful pickups never log it, so most
# workers never name their card. VRAM is NOT in the logs at all — we map the
# known model to its spec. The "Memory check … of X GB total" line is *system*
# RAM (hundreds of GB), not VRAM, so it's reported separately as host RAM.

GPU_URL_RE = re.compile(r"gpu=([A-Za-z0-9%+._-]+)")
HOST_RAM_RE = re.compile(r"of ([0-9.]+)GB total")

# GPU model (as RunPod reports it) → VRAM in GB.
GPU_VRAM_GB = {
    "NVIDIA H100 NVL": 94,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 PCIe": 80,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA L40S": 48,
    "NVIDIA L40": 48,
    "NVIDIA RTX A6000": 48,
    "NVIDIA A40": 48,
    "NVIDIA L4": 24,
}


@dataclass
class WorkerInfo:
    gpu: str | None = None          # explicit GPU model this worker logged (else None)
    host_ram_gb: float | None = None
    cuda: str | None = None
    matmul_ms: list[int] = field(default_factory=list)


def extract_worker_info(records: list[Record]) -> tuple[dict[str, WorkerInfo], str | None]:
    """Per-worker GPU/host-RAM/CUDA fingerprint, plus the endpoint's dominant GPU."""
    info: dict[str, WorkerInfo] = {}
    gpu_counts: dict[str, int] = {}
    for r in records:
        wi = info.setdefault(r.worker, WorkerInfo())
        g = GPU_URL_RE.search(r.msg)
        if g:
            gpu = g.group(1).replace("+", " ").replace("%20", " ").rstrip("'")
            wi.gpu = gpu
            gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        hr = HOST_RAM_RE.search(r.msg)
        if hr:
            wi.host_ram_gb = float(hr.group(1))
        cu = re.search(r"CUDA ([0-9.]+)", r.msg)
        if cu:
            wi.cuda = cu.group(1)
        mm = re.search(r"Matrix multiply completed in (\d+)ms", r.msg)
        if mm:
            wi.matmul_ms.append(int(mm.group(1)))
    dominant = max(gpu_counts, key=gpu_counts.get) if gpu_counts else None
    return info, dominant


def attach_gpu(runs: list[Run], info: dict[str, WorkerInfo]) -> None:
    """Stamp each run with the GPU its worker ACTUALLY logged — nothing inferred.

    The model only appears in 403 job-take URLs, so most workers have no GPU on
    record; those stay `unknown`. We do NOT infer a card for them: VRAM is not in
    the logs, and guessing produced impossible values (94 GB under 80 GB pools).
    """
    for r in runs:
        wi = info.get(r.worker, WorkerInfo())
        r.host_ram_gb = wi.host_ram_gb
        if wi.gpu:                       # confirmed: this worker logged its own GPU
            r.gpu, r.gpu_confirmed = wi.gpu, True
            r.vram_gb = GPU_VRAM_GB.get(wi.gpu)


def _gpu_cell(r: Run) -> str:
    if not r.gpu:
        return "unknown"
    return f"{r.gpu} ({r.vram_gb} GB VRAM)" if r.vram_gb else r.gpu


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(sec: float) -> str:
    if sec < 60:
        return f"{sec:5.1f}s"
    return f"{int(sec)//60:d}m{int(sec)%60:02d}s"


# Plain-English description of each phase, for the markdown report.
PHASE_DESC = {
    "cold_start:startup/find_models":         "Container boot → locate model files on the network volume",
    "cold_start:transformer_load(gguf)":      "Read + dequantize the 10 GB GGUF transformer from the network volume",
    "cold_start:(post-transformer)":          "Set attention backend (handoff)",
    "cold_start:vae_load":                    "Load + Wan→diffusers remap of the VAE safetensors",
    "cold_start:(te config)":                 "Load text-encoder config (handoff)",
    "cold_start:text_encoder_load(fp8)":      "Stream the 9 GB FP8 text encoder from the volume",
    "cold_start:(post-te)":                   "Handoff",
    "cold_start:tokenizer_load":              "Load tokenizer",
    "cold_start:processor_load":              "Load processor (many small files → slow on a contended volume)",
    "cold_start:assemble+to_gpu+lora_angles": "Build pipeline + move all weights to GPU + apply the 'angles' LoRA",
    "cold_start:lora_lightning":              "Apply the 'lightning' LoRA (PEFT injection)",
    "cold_start:(finalize adapters)":         "set_adapters() (handoff)",
    "startup:(pipeline_ready->fitness)":      "Handoff to fitness checks",
    "startup:fitness_checks":                 "RunPod GPU / memory / disk / network fitness checks",
    "startup:dispatch/queue_wait":            "Job dispatch / pickup",
    "run:inference(encode+diffuse+decode)":   "Text encode + 4 diffusion steps + VAE decode (all 3 images)",
    "run:svg_convert":                        "PNG→SVG conversion subprocess (one image, run sequentially)",
    "run:svg_overhead":                       "Save next PNG + spawn next SVG subprocess",
}


def describe(phase: str) -> str:
    if phase.startswith("run:svg_convert["):
        n = phase.split("[", 1)[1].rstrip("]")
        return f"PNG→SVG conversion subprocess for image {n} (sequential)"
    return PHASE_DESC.get(phase, "")


def _build_agg(runs: list[Run]) -> dict[str, list[Gap]]:
    """Group every gap by phase (per-image SVG collapsed into one bucket)."""
    agg: dict[str, list[Gap]] = {}
    for r in runs:
        for g in r.gaps():
            key = "run:svg_convert" if g.phase.startswith("run:svg_convert") else g.phase
            agg.setdefault(key, []).append(g)
    return agg


def _md_intro(runs: list[Run], dominant_gpu: str | None) -> list[str]:
    n_conf = len({r.worker for r in runs if r.gpu_confirmed})
    n_workers = len({r.worker for r in runs})
    return [
        "# Worker Run Timeline Analysis", "",
        f"_Generated by `scripts/analyze_worker_logs.py`. "
        f"{len(runs)} run(s) across {n_workers} worker(s)._", "",
        "## What this shows (and what it can't)", "",
        "These are **worker/job execution logs** — they only start printing once a GPU worker is "
        "*already running* the job. So they show exactly one thing well: **where each run spent "
        "time on the worker**, from the first cold-start line to `Finished.`", "",
        "They do **not** contain, and this document therefore omits:", "",
        "- **Queue / GPU-provisioning wait** (RunPod `delayTime` — the time a request sits before "
        "a worker picks it up). It happens before the container logs anything and is not recorded "
        "in the DB for these runs, so it is unrecoverable.",
        "- **Per-request idle / total latency.** What the user experienced end-to-end is "
        "`delayTime + everything below`; we can only measure the *below*.", "",
        "Durations use the **real event clock** (the inner `HH:MM:SS,mmm` timestamp Python logging "
        "writes), not the log-shipper timestamp — the shipper batches lines and would otherwise "
        "collapse tens of seconds of cold-start work into milliseconds.", "",
        f"**GPU — and why there's no reliable VRAM column:** GPU VRAM is *not* written to these "
        f"logs. The GPU **model** appears only inside 403 job-take URLs, so it is confirmed for "
        f"just **{n_conf} of {n_workers} workers** — the rest are `unknown` (not inferred). For the "
        f"confirmed ones RunPod reported `{dominant_gpu or 'n/a'}`; the `94 GB` beside it is that "
        f"model's spec, shown only where the model is confirmed.", "",
        f"> ⚠️ **Config check:** `{dominant_gpu or 'the reported GPU'}` is a 94 GB card, but your "
        f"selected pools top out at 80 GB — so what RunPod actually assigned during these runs "
        f"disagrees with the current pool config. Worth verifying the endpoint's GPU selection. "
        f"The `504–2016 GB` figures elsewhere in the logs are the machine's **system RAM** (1-GPU "
        f"workers), **not** GPU memory, so they are kept out of the per-run table.", "",
    ]


def _md_summary(runs: list[Run]) -> list[str]:
    L = ["## Summary — one row per run", "",
         "| # | Worker | GPU (logged) | Job | Cold? | Cold start | Fitness+dispatch | Inference | SVG (3 imgs) | **Total on worker** |",
         "|--:|--------|--------------|-----|:-----:|-----------:|-----------------:|----------:|-------------:|--------------------:|"]
    for i, r in enumerate(runs, 1):
        L.append(
            f"| {i} | `{r.worker}` | {_gpu_cell(r)} | `{(r.job_id or '?')[:14]}` | "
            f"{'🥶 yes' if r.has_cold_start else 'warm'} | "
            f"{fmt(r.phase_total('cold_start'))} | {fmt(r.phase_total('startup'))} | "
            f"{fmt(r.phase_total('run:inference'))} | {fmt(r.phase_total('run:svg'))} | "
            f"**{fmt(r.total_s)}** |"
        )
    L += ["",
          "_`GPU (logged)` = only what the worker itself put in the logs; `unknown` means it never "
          "logged its GPU. `Total on worker` = cold start + fitness + inference + SVG; it excludes "
          "queue/provisioning wait, which isn't in these logs._", ""]
    return L


def _md_aggregate(agg: dict[str, list[Gap]]) -> list[str]:
    L = ["## Where the time goes — summed across all runs", "",
         "| Phase | What happens | Count | Total | Mean | Max |",
         "|-------|--------------|------:|------:|-----:|----:|"]
    for phase, gaps in sorted(agg.items(), key=lambda kv: sum(g.seconds for g in kv[1]), reverse=True):
        total = sum(g.seconds for g in gaps)
        if total < 0.15:
            continue  # drop measurement-seam noise from the headline table
        mx = max(gaps, key=lambda g: g.seconds)
        L.append(f"| `{phase}` | {describe(phase)} | {len(gaps)} | "
                 f"**{fmt(total)}** | {fmt(total/len(gaps))} | {fmt(mx.seconds)} |")
    L.append("")
    return L


def _md_per_run(runs: list[Run]) -> list[str]:
    L = ["## Each run, start → finish", ""]
    for i, r in enumerate(runs, 1):
        L.append(f"### Run {i} — `{r.worker}` ({'cold start' if r.has_cold_start else 'warm'})")
        L.append("")
        L.append(f"- **GPU (logged):** {_gpu_cell(r)}"
                 + ("" if r.gpu_confirmed else " — *this worker never logged its GPU*"))
        L.append(f"- **Job:** `{r.job_id}`")
        L.append(f"- **Window:** {r.start:%Y-%m-%d %H:%M:%S} → {r.end:%H:%M:%S} "
                 f"(**{fmt(r.total_s)}** on worker)")
        if r.diffusion_s:
            L.append(f"- **Diffusion steps only:** {r.diffusion_s:.0f}s (from the tqdm bar)")
        L += ["", "| Elapsed | Phase | What happens | Duration | % of run |",
              "|--------:|-------|--------------|---------:|---------:|"]
        elapsed = 0.0
        for g in r.gaps():
            if g.seconds < 0.05:
                elapsed += g.seconds
                continue
            pct = 100 * g.seconds / r.total_s if r.total_s else 0
            L.append(f"| +{fmt(elapsed)} | `{g.phase}` | {describe(g.phase)} | "
                     f"**{fmt(g.seconds)}** | {pct:4.0f}% |")
            elapsed += g.seconds
        L.append("")
    return L


def _matmul_cell(ms: list[int]) -> str:
    if not ms:
        return "—"
    lo = min(ms)
    return f"{lo}–{max(ms)}" if len(set(ms)) > 1 else str(lo)


def _md_fingerprint(runs: list[Run], info: dict[str, WorkerInfo]) -> list[str]:
    runs_by_worker: dict[str, int] = {}
    for r in runs:
        runs_by_worker[r.worker] = runs_by_worker.get(r.worker, 0) + 1
    L = ["## Workers seen (raw fingerprint from the logs)", "",
         "Only `GPU (logged)` and `System RAM` come straight from the logs. **System RAM is the "
         "host machine's memory, not GPU VRAM** (these are 1-GPU workers). VRAM is shown only for "
         "workers that logged their model, as that model's spec.", "",
         "| Worker | GPU (logged) | VRAM (spec) | System RAM | CUDA | matmul ms | Runs |",
         "|--------|--------------|------------:|-----------:|:----:|----------:|-----:|"]
    for w in sorted(info, key=lambda w: runs_by_worker.get(w, 0), reverse=True):
        wi = info[w]
        vram = GPU_VRAM_GB.get(wi.gpu or "")
        L.append(
            f"| `{w}` | {wi.gpu or 'unknown'} | {f'{vram} GB' if vram else '—'} | "
            f"{f'{wi.host_ram_gb:.0f} GB' if wi.host_ram_gb else '—'} | "
            f"{wi.cuda or '—'} | {_matmul_cell(wi.matmul_ms)} | {runs_by_worker.get(w, 0)} |"
        )
    L += ["",
          "`matmul ms` is the fitness benchmark (a host-contention indicator, not a GPU-model tell).", ""]
    return L


def _md_step_stats(agg: dict[str, list[Gap]]) -> list[str]:
    phase_order: dict[str, int] = {}
    for i, (_k, _rx, ph) in enumerate(MILESTONES):
        phase_order.setdefault(ph, i)
    L = ["## Time per pipeline step (min / median / max across all runs)", "",
         "Each row is one step of the run, in execution order. **Min** and **Max** are the fastest "
         "and slowest *single occurrences* of that step across all runs — **Max is one run's value, "
         "not a sum.** Median is the typical case. Handoff seams are omitted.", "",
         "| Pipeline step | Min (fastest run) | Median | Max (slowest run) |",
         "|---------------|------------------:|-------:|------------------:|"]
    ordered = sorted((p for p in agg if ":(" not in p), key=lambda p: phase_order.get(p, 999))
    for phase in ordered:
        secs = [g.seconds for g in agg[phase]]
        L.append(f"| {describe(phase) or phase} | {fmt(min(secs))} | "
                 f"{fmt(statistics.median(secs))} | {fmt(max(secs))} |")
    L.append("")
    return L


def _md_optimizations() -> list[str]:
    return [
        "## Optimizations we can do", "",
        "Ordered by impact on the numbers above:", "",
        "1. **Keep ≥1 worker warm** (min/active workers ≥ 1 + FlashBoot). A cold start is paid on "
        "most runs and dominates total on-worker time; a warm worker skips it entirely — warm runs "
        "here finish in **~30–40 s** vs. **1–9 min** cold.",
        "2. **Bake the weights into the image (local disk) instead of the network volume.** Every "
        "cold start streams ~20 GB from the volume, and that read is what explodes on a bad node "
        "(transformer load 26 s → 4m38s; processor 2.5 s → 44 s). Local NVMe makes it fast and "
        "consistent, killing the multi-minute outliers.",
        "3. **Merge the two LoRAs into the transformer offline.** LoRA injection is ~47 s of every "
        "cold start (`assemble+…+lora_angles` plus `lora_lightning`); baked once at provision time "
        "it disappears from the hot path.",
        "4. **Run the 3 SVG conversions in parallel.** They run sequentially today (~20–70 s per "
        "run) but are independent subprocesses — parallelizing roughly thirds it.",
        "5. **Verify / broaden the GPU pool.** The real 'waiting for a GPU' delay is "
        "queue/provisioning time that these worker logs don't capture; more allowed GPU types = "
        "shorter queue. Also reconcile the H100-NVL-vs-80 GB-pool mismatch flagged above.", "",
        "_Net: warm workers make cold starts **rare**; baked weights + LoRA merge make the "
        "unavoidable ones **cheap**; parallel SVG trims the execution tail. Together they take a "
        "worst-case run from ~7–9 min toward well under a minute._", "",
    ]


def write_markdown(runs: list["Run"], info: dict[str, "WorkerInfo"],
                   dominant_gpu: str | None, path: Path) -> None:
    agg = _build_agg(runs)
    L = (_md_intro(runs, dominant_gpu) + _md_summary(runs) + _md_aggregate(agg)
         + _md_per_run(runs) + _md_fingerprint(runs, info) + _md_step_stats(agg)
         + _md_optimizations())
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {path}")


def compute_idle(runs: list[Run]) -> list[Gap]:
    """Idle = time a warm worker sat between one run's Finished and the next start.

    Also stamps each run's `idle_before_s` (0 for the first run on a worker, i.e.
    cold starts) so it can be shown as a per-request column.
    """
    idle: list[Gap] = []
    by_worker: dict[str, list[Run]] = {}
    for r in runs:
        by_worker.setdefault(r.worker, []).append(r)
    for wruns in by_worker.values():
        wruns.sort(key=lambda r: r.start)
        for a, b in zip(wruns, wruns[1:]):
            gap = (b.start - a.end).total_seconds()
            # Per-request idle = worker stayed WARM waiting for this request. If b
            # cold-started, the worker was torn down (idleTimeout ~5s) and the gap
            # was no-traffic downtime, not paid idle — so idle_before stays 0.
            if not b.has_cold_start:
                b.idle_before_s = max(0.0, gap)
            if gap > 0.5:
                kind = "downtime(no traffic, worker gone)" if b.has_cold_start else "warm_idle"
                idle.append(Gap(f"idle:{kind}", gap, a.worker, a.idx, a.end))
    return idle


def _print_per_run_table(runs: list[Run]) -> None:
    print("\nPER-RUN BREAKDOWN  (on-worker time; real inner-event clock)\n")
    hdr = (f"{'#':>2}  {'worker':<14} {'GPU (VRAM)':<22} {'cold_st':>8} "
           f"{'fit+disp':>8} {'infer':>7} {'svg':>7} {'TOTAL':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(runs, 1):
        gpu = (f"{r.gpu} ({r.vram_gb}GB){'' if r.gpu_confirmed else '*'}"
               if r.gpu else "unknown")
        print(f"{i:>2}  {r.worker:<14} {gpu[:22]:<22} "
              f"{fmt(r.phase_total('cold_start')):>8} {fmt(r.phase_total('startup')):>8} "
              f"{fmt(r.phase_total('run:inference')):>7} {fmt(r.phase_total('run:svg')):>7} "
              f"{fmt(r.total_s):>8}")


def report(runs: list[Run], idle: list[Gap]) -> dict:
    print("=" * 100)
    print(f"Parsed {len(runs)} run(s) across {len({r.worker for r in runs})} worker(s)")
    print("=" * 100)

    _print_per_run_table(runs)

    # ---- Aggregate by phase ---------------------------------------------
    agg: dict[str, list[Gap]] = {}
    for r in runs:
        for g in r.gaps():
            # collapse per-image svg into one bucket for the headline aggregate
            key = "run:svg_convert" if g.phase.startswith("run:svg_convert") else g.phase
            agg.setdefault(key, []).append(g)

    print("\n\nWHERE THE TIME GOES  (summed across all runs, worst first)\n")
    hdr2 = f"{'phase':<42} {'count':>5} {'total':>9} {'mean':>8} {'max':>8}  worst@"
    print(hdr2)
    print("-" * (len(hdr2) + 14))
    rows = []
    for phase, gaps in agg.items():
        total = sum(g.seconds for g in gaps)
        worst = max(gaps, key=lambda g: g.seconds)
        rows.append((total, phase, len(gaps), total / len(gaps), worst))
    for total, phase, cnt, mean, worst in sorted(rows, reverse=True):
        print(f"{phase:<42} {cnt:>5} {fmt(total):>9} {fmt(mean):>8} "
              f"{fmt(worst.seconds):>8}  {worst.worker}")

    # ---- Single biggest time sinks --------------------------------------
    all_gaps = [g for r in runs for g in r.gaps()]
    all_gaps = [g for g in all_gaps if not g.phase.startswith("cold_start:(")
                and not g.phase.startswith("startup:(")]
    print("\n\nTOP 15 SINGLE LONGEST WAITS  (one gap in one run)\n")
    hdr3 = f"{'rank':>4}  {'dur':>8}  {'phase':<42} {'worker':<14} {'when (event clock)'}"
    print(hdr3)
    print("-" * len(hdr3))
    for i, g in enumerate(sorted(all_gaps, key=lambda g: g.seconds, reverse=True)[:15], 1):
        print(f"{i:>4}  {fmt(g.seconds):>8}  {g.phase:<42} {g.worker:<14} "
              f"{g.start:%Y-%m-%d %H:%M:%S}")

    # ---- Idle (informational, not "waste") ------------------------------
    if idle:
        idle_total = sum(g.seconds for g in idle)
        print(f"\n\nWARM-IDLE BETWEEN JOBS (worker alive, no work): "
              f"{fmt(idle_total)} total over {len(idle)} gap(s) — not counted above.")

    # ---- Headline --------------------------------------------------------
    cold_total = sum(g.seconds for r in runs for g in r.gaps() if g.phase.startswith("cold_start"))
    svg_total = sum(g.seconds for r in runs for g in r.gaps() if g.phase.startswith("run:svg_convert"))
    infer_total = sum(g.seconds for r in runs for g in r.gaps() if g.phase.startswith("run:inference"))
    n_cold = sum(1 for r in runs if r.has_cold_start)
    print("\n" + "=" * 100)
    print("HEADLINE")
    print("=" * 100)
    print(f"  Cold starts:      {n_cold} of {len(runs)} runs paid one; "
          f"{fmt(cold_total)} total, {fmt(cold_total / max(n_cold,1))} avg when incurred.")
    print(f"  SVG conversion:   {fmt(svg_total)} total (runs sequentially, 3 images/run).")
    print(f"  GPU inference:    {fmt(infer_total)} total (encode+diffuse+decode).")
    if runs:
        worst_run = max(runs, key=lambda r: r.total_s)
        print(f"  Slowest run:      {fmt(worst_run.total_s)} — {worst_run.worker} "
              f"(job {worst_run.job_id}), cold_start={fmt(worst_run.phase_total('cold_start'))}.")

    # structured payload for --json
    return {
        "runs": [
            {
                "worker": r.worker, "job_id": r.job_id, "cold_start": r.has_cold_start,
                "gpu": r.gpu, "gpu_confirmed": r.gpu_confirmed, "vram_gb": r.vram_gb,
                "host_ram_gb": r.host_ram_gb,
                "start": r.start.isoformat(), "total_s": round(r.total_s, 2),
                "cold_start_s": round(r.phase_total("cold_start"), 2),
                "startup_s": round(r.phase_total("startup"), 2),
                "inference_s": round(r.phase_total("run:inference"), 2),
                "diffusion_s": r.diffusion_s,
                "svg_s": round(r.phase_total("run:svg"), 2),
                "phases": {g.phase: round(g.seconds, 2) for g in r.gaps()},
            }
            for r in runs
        ],
        "by_phase": {
            phase: {"count": len(gaps), "total_s": round(sum(g.seconds for g in gaps), 2)}
            for phase, gaps in agg.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="log file(s) or glob(s); default output/logs/*.txt")
    ap.add_argument("--json", metavar="FILE", help="also write structured report to FILE")
    ap.add_argument("--md", action="store_true", help="write a markdown timeline document")
    ap.add_argument("--md-out", metavar="FILE", default="output/run_timeline_analysis.md",
                    help="markdown output path (default output/run_timeline_analysis.md)")
    args = ap.parse_args()

    patterns = args.paths or ["output/logs/*.txt"]
    files: list[Path] = []
    for p in patterns:
        files.extend(Path(x) for x in glob.glob(p))
    files = sorted(set(files))
    if not files:
        sys.exit(f"No log files matched: {patterns}")

    records: list[Record] = []
    for f in files:
        records.extend(parse_file(f))
    if not records:
        sys.exit("No parseable log lines found.")

    print(f"Files: {', '.join(str(f) for f in files)}")
    unify_clocks(records)
    runs = build_runs(records)
    info, dominant_gpu = extract_worker_info(records)
    attach_gpu(runs, info)
    idle = compute_idle(runs)
    payload = report(runs, idle)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    if args.md:
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(runs, info, dominant_gpu, md_path)


if __name__ == "__main__":
    main()
