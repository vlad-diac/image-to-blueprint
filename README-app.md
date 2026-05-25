# App — Local Dev

NestJS API ([apps/api/](apps/api/)) + React/Vite client ([apps/web/](apps/web/)) backed by Postgres. The API submits jobs to a deployed RunPod endpoint via `/run`, stores each `Run` (prompts, settings, input/output PNG bytes, `runpodJobId`, RunPod timings) in Postgres, and reconciles status by calling `/status` lazily on `GET /runs/:id` plus a small `@nestjs/schedule` sweep. The web client uploads an image, polls the run, and renders a side-by-side input/output preview with an expandable details panel. No queue, no Redis — single-user POC. See [docs/poc-implementation-plan.md](docs/poc-implementation-plan.md) for full design.

## Prerequisites

- Node 20+, `pnpm`
- Docker (for Postgres)
- A deployed RunPod endpoint — see [README-worker.md](README-worker.md)

## Run

1. **Install deps** (from repo root):
   ```bash
   pnpm install
   ```

2. **Start Postgres**:
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```
   Listens on `localhost:5454` (user/pass `postgres`, db `blueprint`).

3. **Configure the API** — create [apps/api/.env](apps/api/.env):
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5454/blueprint
   RUNPOD_API_KEY=<your runpod api key>
   RUNPOD_ENDPOINT_ID=<your endpoint id>
   PORT=3001
   ```

4. **Apply Prisma schema**:
   ```bash
   pnpm --filter api prisma:migrate
   ```

5. **Configure the web client** — create [apps/web/.env](apps/web/.env):
   ```env
   VITE_API_URL=http://localhost:3001
   ```

6. **Run both apps** (from repo root):
   ```bash
   pnpm dev
   ```
   - API: http://localhost:3001
   - Web: http://localhost:5173 (Vite default)

   Or individually: `pnpm dev:api` / `pnpm dev:web`.

7. **Use it** — open the web UI, drop an image, set the prompt, click *Process*. The client polls `GET /runs/:id` every ~1.5 s while the run is `QUEUED` / `IN_QUEUE` / `IN_PROGRESS`; the output PNG renders once status flips to `SUCCEEDED`.

## Key endpoints (API)

- `POST /runs` — multipart upload, submits to RunPod, returns `{ id, runpodJobId, status }`
- `GET /runs/:id` — full record (reconciles via `/status` when in-flight)
- `GET /runs/:id/input.png` / `GET /runs/:id/output.png` — image bytes
- `GET /runs` — recent runs (no blobs)
- `POST /runs/:id/cancel` — calls RunPod `/cancel/<id>`
