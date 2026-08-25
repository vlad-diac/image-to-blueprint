# scripts/ — local ops tools (not deployed)

Scripts here run **on your machine**, not inside any container. They are
deliberately kept **out of `worker/scripts/`** because the worker image bundles
that directory wholesale (`COPY worker/scripts/ /app/scripts/` in
[worker/Dockerfile](../worker/Dockerfile)). Nothing in this folder is copied
into the image or referenced by `docker-compose.dev.yml`.

## Scripts

- [`check_gpu_availability.py`](check_gpu_availability.py) — check RunPod GPU
  availability per datacenter/region, grouped by GPU pool
  (`runpodctl datacenter list`). Requires `runpodctl` installed + authenticated.

  Each run saves a timestamped snapshot as its own file in a history folder
  (`scripts/gpu_availability_history/`, **committed to git** so runs can be
  compared over time). On every run the script loads **all** snapshot files in
  that folder, merges them, and:

  - prints an **availability-by-time-of-day** summary — average stock per pool
    bucketed into local-time periods (Night / Morning / Afternoon / Evening), so
    repeated polling reveals *when* each pool tends to have stock; and
  - (re)writes the merged **official list** — `official.json` — the consolidated
    view (per-pool time-of-day + best-ever pick) you compare individual runs
    against.

  ```
  scripts/gpu_availability_history/
    run-20260825T110729Z.json   # one file per run (UTC timestamp)
    run-20260825T110747Z.json
    official.json               # merged/consolidated result, regenerated each run
  ```

  ```bash
  python scripts/check_gpu_availability.py                  # default pools
  python scripts/check_gpu_availability.py ADA_80_PRO --region Europe
  python scripts/check_gpu_availability.py all --min-stock Medium
  python scripts/check_gpu_availability.py --list-pools

  python scripts/check_gpu_availability.py --history-dir /tmp/gpu-history
  python scripts/check_gpu_availability.py --no-save        # don't record this run
  python scripts/check_gpu_availability.py --no-time-of-day # skip the summary
  ```

  Run it on a schedule (e.g. cron) to build up history, then commit the folder:

  ```bash
  */30 * * * * cd /path/to/image-to-blueprint && python scripts/check_gpu_availability.py all >/dev/null 2>&1
  ```

- [`copy_network_volume.py`](copy_network_volume.py) — copy the `/workspace` of one
  RunPod network volume into another. A pod can attach only one volume, so this
  spins up one pod per volume and transfers pod-to-pod (`rsync` over SSH by default,
  or `runpodctl send`/`receive` with `--method send`), then verifies and tears the
  pods down. Requires `runpodctl` authenticated and your SSH key registered
  (`runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`).

  ```bash
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST
  python scripts/copy_network_volume.py --source-volume-id SRC \
      --create-dest --dest-datacenter EU-RO-1 --size 60
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST --dry-run
  ```
