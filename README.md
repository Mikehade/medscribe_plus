# medscribe_plus
Medscribe

# MedScribe+

> AI-powered medical scribe — real-time audio transcription, structured SOAP note generation, clinical quality evaluation, and EHR insertion.

---

## Table of Contents

- [Overview](#overview)
- [Monorepo Structure](#monorepo-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Running the Full Stack](#running-the-full-stack)
- [Running Backend Only](#running-backend-only)
- [Running Frontend Only](#running-frontend-only)
- [Local Development (No Docker)](#local-development-no-docker)
- [Testing](#testing)
- [Makefile Reference](#makefile-reference)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Overview

MedScribe+ automates clinical documentation end-to-end:

1. **Transcription** — Live or uploaded audio is transcribed via AWS Nova 2 Sonic
2. **SOAP Generation** — A ScribeAgent produces structured SOAP notes via AWS Bedrock
3. **Clinical Evaluation** — Hallucination detection, drug interaction checks, and guideline alignment run in parallel
4. **EHR Insertion** — Physician approves the note; Nova Act inserts it into the EHR

**Stack:**
- Backend: FastAPI + Python 3.11 + Redis + ChromaDB
- Frontend: React + TypeScript + Vite
- Infrastructure: AWS Bedrock, Nova Act, Redis

---

## Monorepo Structure

```
medscribe_plus/
├── Makefile                  ← Root Makefile — manages full stack and backend+frontend separately
├── docker-compose.yml        ← Full stack compose (Redis + API + Frontend)
├── scripts/
│   └── start.sh              ← Single-command starter script
│
├── backend/                  ← FastAPI backend (can run standalone)
│   ├── README.md             ← Backend-specific setup and instructions ← READ THIS
│   ├── Makefile              ← Backend-only commands
│   ├── docker-compose.yml    ← Backend-only compose (Redis + API)
│   ├── Dockerfile
│   ├── conftest.py
│   ├── requirements.txt
│   ├── main.py
│   ├── src/
│   ├── tests/
│   └── utils/
│
└── frontend/                 ← React/Vite frontend
    ├── Dockerfile
    ├── nginx.conf
    ├── src/
    └── package.json
```

> **Backend-only instructions** → see [`backend/README.md`](./backend/README.md)

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker | 24+ | Required for containerised runs |
| Docker Compose | v2 (`docker compose`) | Bundled with Docker Desktop |
| GNU Make | 3.81+ | Pre-installed on macOS/Linux |
| Node.js | 20+ | Only needed for local frontend dev |
| Python | 3.11+ | Only needed for local backend dev |
| AWS account | — | Bedrock + Nova Act access required |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/medscribe_plus.git
cd medscribe_plus

# 2. Set up environment
cp backend/env.example backend/.env
# Edit backend/.env — add AWS keys, Redis password, etc.

# 3. Create root .env for Docker Compose variable substitution
grep -E "^REDIS_|^APP_ENV|^LOG_LEVEL" backend/.env > .env

# 4a. Start backend only (API + Redis)
make start-backend

# 4b. OR start the full stack (API + Redis + Frontend)
make start
```

After `make start`:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Environment Setup

All configuration lives in `backend/.env`. A root `.env` is also required for Docker Compose variable substitution.

### Step 1 — Create backend config

```bash
cp backend/env.example backend/.env
```

Fill in these required values:

| Variable | Description |
|----------|-------------|
| `APP_ENV` | `development` / `staging` / `production` |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |
| `AWS_ACCESS_KEY` | AWS IAM access key ID |
| `AWS_SECRET_KEY` | AWS IAM secret access key |
| `AWS_REGION_NAME` | AWS region — Sonic requires `us-east-1` |
| `NOVA_ACT_API_KEY` | Nova Act API key |
| `SERP_API_KEY` | SERP API key |
| `REDIS_HOST` | `127.0.0.1` (Docker) or `localhost` (local) |
| `REDIS_PORT` | `6379` |
| `REDIS_PASSWORD` | Your Redis password |
| `REDIS_URL` | `redis://:password@127.0.0.1:6379/0` |
| `REDIS_LOCATION` | `redis://:password@127.0.0.1:6379/0` |

> ⚠️ `REDIS_URL` format: `:password@host` — **no username before the colon**

### Step 2 — Create root .env for Docker Compose

Docker Compose resolves `${VARIABLE}` substitutions from a `.env` file in the **same directory as the compose file** (the project root). This file only needs the Redis vars:

```bash
grep -E "^REDIS_|^APP_ENV|^LOG_LEVEL" backend/.env > .env
```

Never commit either `.env` file — both are gitignored.

---

## Running the Full Stack

Starts Redis + API + Frontend together. Frontend proxies all `/api/*` requests to the backend via nginx — no CORS issues.

```bash
make start
```

Or with the shell script:
```bash
bash scripts/start.sh full
```

**Stop:**
```bash
make stop
```

**Restart:**
```bash
make restart
```

**View logs:**
```bash
make logs           # all services
make logs-api       # API only
make logs-redis     # Redis only
make logs-frontend  # Frontend/nginx only
```

---

## Running Backend Only

The backend can run completely independently — useful during API development or when the frontend is served separately.

```bash
# From the root
make start-backend

# Or cd into backend and use its own Makefile
cd backend
make start        # uses backend/docker-compose.yml
```

> See **[`backend/README.md`](./backend/README.md)** for full backend-specific instructions including local dev setup, testing, and all backend Makefile targets.

---

## Running Frontend Only

Requires the backend to already be running.

```bash
bash scripts/start.sh frontend
```

Or rebuild and restart just the frontend container:
```bash
make build-frontend
docker compose --profile full up -d frontend
```

For local Vite dev server (hot-reload, no Docker):
```bash
make run-frontend-local
# Runs on http://localhost:3000
# Proxies /api/* to http://localhost:8000 via vite.config.ts proxy
```

---

## Local Development (No Docker)

Run both services locally with hot-reload — no Docker required.

```bash
# Terminal 1 — start Redis (Docker still needed for Redis)
make run-redis-local

# Terminal 2 — start backend API
make run-backend-local

# Terminal 3 — start frontend Vite dev server
make run-frontend-local
```

| Service | URL |
|---------|-----|
| Frontend (Vite) | http://localhost:3000 |
| API (uvicorn) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Testing

Tests live in `backend/tests/`. See [`backend/README.md`](./backend/README.md) for full testing documentation.

```bash
# Run all backend unit tests
make test

# Run with HTML coverage report
make test-cov
# Opens backend/htmlcov/index.html
```

---

## Makefile Reference

Run `make help` to see all targets. Key ones:

| Target | Description |
|--------|-------------|
| `make start` | Build + start full stack (Redis + API + Frontend) |
| `make start-backend` | Build + start backend only (Redis + API) |
| `make stop` | Stop full stack |
| `make restart` | Stop then start full stack |
| `make build` | Build all Docker images |
| `make build-backend` | Build backend image only |
| `make build-frontend` | Build frontend image only |
| `make logs` | Tail all container logs |
| `make logs-api` | Tail API logs |
| `make logs-frontend` | Tail nginx/frontend logs |
| `make logs-redis` | Tail Redis logs |
| `make shell-api` | bash shell inside API container |
| `make test` | Run backend unit tests |
| `make test-cov` | Run tests with coverage report |
| `make run-backend-local` | Start API locally with hot-reload |
| `make run-frontend-local` | Start Vite dev server |
| `make run-redis-local` | Start local Redis container |
| `make ingest` | Ingest clinical PDFs into ChromaDB |
| `make clean` | Remove build artifacts |
| `make clean-all` | Remove everything including venv and node_modules |

---

## Architecture

```
Browser
  │
  │  HTTPS (Codespaces) or HTTP (local)
  ▼
nginx :3000
  │
  ├── /assets/*    → static files (SPA bundle)
  ├── /api/*       → proxy → FastAPI :8000
  │                   (wss:// → ws:// upgrade for WebSocket)
  └── /*           → index.html (SPA routing)
                         │
                    FastAPI :8000
                         │
              ┌──────────┼──────────┐
              │          │          │
           Redis      ChromaDB   AWS Bedrock
         (cache,      (RAG /     (LLM + Sonic
        sessions)    vectors)    transcription)
```

**Key design decisions:**
- Both containers use `network_mode: host` for Codespaces compatibility — no port forwarding layer
- nginx handles TLS termination for WebSocket: browser sends `wss://`, nginx proxies plain `ws://` to backend
- Frontend is built with `VITE_API_URL=/api/v1` baked in — all API calls are relative and go through nginx
- Redis auth uses `:password@host` format — no username (Redis default config has no ACL users)

---

## Troubleshooting

### `make start` fails — API unhealthy
```bash
make logs-api
# Most common causes: Redis connection error, missing env var, import error
```

### `curl localhost:8000/health` fails in Codespaces
Both containers use `network_mode: host` — they bind directly to the VM network. If curl fails, the container crashed. Check `make logs-api`.

### Redis auth error — `invalid username-password pair`
Your `REDIS_URL` or `REDIS_LOCATION` has a username (`default:password`). Remove the username:
```bash
# Wrong
redis://default:password@127.0.0.1:6379/0

# Correct
redis://:password@127.0.0.1:6379/0
```

### 413 Request Entity Too Large
nginx is configured for 100MB uploads. If you still see this, the request is hitting the backend directly (not through nginx). Use the frontend URL (port 3000) not the API URL (port 8000) for uploads.

### WebSocket error — `insecure WebSocket connection from HTTPS`
The frontend WS URL must use `wss://` when the page is served over HTTPS. Update `LiveRecording.tsx`:
```typescript
const WS_URL = import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}://${window.location.host}/api/transcribe/`;
```

### Frontend not picking up API URL changes
`VITE_API_URL` is baked into the bundle at build time. After changing `frontend/.env.production`, you must rebuild:
```bash
make build-frontend
docker compose --profile full up -d frontend
```

### Module errors during test collection
The root `conftest.py` stubs heavy SDK modules. If you see a new `ModuleNotFoundError`, add the module to `conftest.py`. See [`backend/README.md`](./backend/README.md) for details.