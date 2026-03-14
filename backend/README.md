# MedScribe+

> AI-powered medical scribe — real-time transcription, structured SOAP note generation, clinical evaluation, and EHR insertion.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Local Development Setup](#local-development-setup)
- [Running the API Locally](#running-the-api-locally)
- [Running with Docker](#running-with-docker)
- [Running with Docker Compose (recommended)](#running-with-docker-compose-recommended)
- [Testing](#testing)
- [Makefile Reference](#makefile-reference)
- [API Endpoints](#api-endpoints)
- [Ingesting Clinical Documents](#ingesting-clinical-documents)
- [Troubleshooting](#troubleshooting)

---

## Overview

MedScribe+ is a FastAPI backend that orchestrates a multi-agent pipeline for automated clinical documentation:

1. **Transcription** — Audio from a live consultation or uploaded file is transcribed via AWS Nova 2 Sonic (bidirectional streaming).
2. **SOAP Generation** — The transcript is processed by a ScribeAgent that calls an LLM (AWS Bedrock) to produce a structured SOAP note.
3. **Clinical Evaluation** — An EvaluationAgent runs hallucination detection, drug interaction checks, and clinical guideline alignment in parallel.
4. **EHR Insertion** — The physician reviews the note in a dashboard and approves it for insertion into the EHR via Nova Act browser automation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Scribe API  │  │   SOAP API   │  │      RAG API         │  │
│  │  (WebSocket) │  │  (REST)      │  │  (REST)              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│  ┌──────▼─────────────────▼──────────────────────▼───────────┐ │
│  │                     ScribeAgent                            │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ SOAPTools   │  │ PatientTools │  │ScribeEvalTools   │ │ │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘ │ │
│  │         │                │                   │            │ │
│  │  ┌──────▼──────┐  ┌──────▼───────┐  ┌────────▼─────────┐│ │
│  │  │ SOAPService │  │PatientService│  │ EvaluationAgent  ││ │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘│ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Infrastructure Layer                    │   │
│  │  BedrockModel  │  SonicModel  │  CacheService  │  RAG    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
    AWS Bedrock          Redis Cache          ChromaDB
    (LLM + Sonic)      (transcripts,         (clinical
                         SOAP, scores)        documents)
```

---

## Project Structure

```
medscribe_plus/backend/
│
├── conftest.py                     # Pytest root conftest — stubs uninstallable SDKs
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Full stack: API + Redis
├── Makefile                        # All dev, test, and Docker commands
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
├── main.py                         # FastAPI app entrypoint
├── pytest.ini                      # Pytest configuration
│
├── scripts/
│   ├── check_env.sh                # Validates required env vars before build/run
│   └── run_tests.sh                # Flexible test runner (unit/cov/ci/file modes)
│
├── src/
│   ├── api/                        # Route handlers and WebSocket consumers
│   │   ├── base/                   # Health check and base routes
│   │   ├── rag/                    # RAG document retrieval endpoints
│   │   ├── scribe/                 # Real-time scribe WebSocket + REST endpoints
│   │   └── soap/                   # SOAP note retrieval endpoints
│   │
│   ├── config/                     # Pydantic settings and DI container
│   │   ├── base.py                 # Settings class (reads from .env)
│   │   ├── dependency_injection/
│   │   │   └── container.py        # Wires all services, tools, and agents
│   │   ├── development.py
│   │   ├── staging.py
│   │   └── production.py
│   │
│   ├── core/
│   │   ├── agents/                 # LLM agent orchestrators
│   │   │   ├── base.py             # Abstract BaseAgent + ensure_async_generator
│   │   │   ├── scribe.py           # ScribeAgent — full consultation pipeline
│   │   │   ├── evaluation.py       # EvaluationAgent — clinical quality checks
│   │   │   └── nova_act.py         # NovaActAgent — EHR browser automation
│   │   │
│   │   ├── prompts/                # System prompt templates
│   │   │   ├── base.py
│   │   │   ├── scribe.py
│   │   │   └── evaluation.py
│   │   │
│   │   └── tools/                  # Agent-callable tool wrappers
│   │       ├── base.py             # BaseTool — get_tool_method, kwargs injection
│   │       ├── soap.py             # SOAPTools — generate_soap_note, get_transcript
│   │       ├── patient.py          # PatientTools — EHR history, insert, flag fields
│   │       ├── evaluation.py       # EvaluationTools — hallucination, drug, guideline
│   │       ├── retriever.py        # RetrieverTools — RAG semantic search
│   │       └── scribe_evaluation.py# ScribeEvaluationTools — bridge to EvaluationAgent
│   │
│   └── infrastructure/
│       ├── cache/                  # Redis-backed cache abstraction
│       │   ├── base.py
│       │   ├── service.py          # CacheService facade
│       │   └── redis/
│       │       ├── client.py
│       │       └── manager.py
│       │
│       ├── embedding_models/       # Text embedding abstractions
│       │   ├── base.py
│       │   └── bedrock.py          # BedrockEmbeddingModel
│       │
│       ├── language_model_service/ # High-level LLM service (text/image/doc)
│       │   └── bedrock.py          # BedrockModelService
│       │
│       ├── language_models/        # Low-level LLM wrappers
│       │   ├── base.py             # BaseLLMModel — template rendering, history
│       │   ├── bedrock.py          # BedrockModel — AWS Bedrock converse API
│       │   └── sonic.py            # SonicModel — Nova 2 Sonic bidirectional stream
│       │
│       ├── services/               # Domain services (no agent/tool logic)
│       │   ├── soap.py             # SOAPService — generate, cache, retrieve
│       │   ├── evaluation.py       # EvaluationService — hallucination, drug, guideline
│       │   ├── patient.py          # PatientService — EHR history, insert, flag fields
│       │   ├── rag.py              # RAGService — retrieve, ingest
│       │   └── transcription.py    # TranscriptionService — file upload + real-time
│       │
│       └── vector_store/           # ChromaDB vector store abstraction
│           ├── base.py
│           └── chroma.py
│
├── utils/
│   ├── helpers.py                  # PDF/DOCX utilities, JSON fixers
│   ├── ingest.py                   # Document chunking and ingestion pipeline
│   └── logger.py                   # Structured logger setup
│
├── documents/                      # Clinical reference PDFs (for RAG ingestion)
│   ├── Clinical Guideline.pdf
│   ├── Drug Interaction Reference.pdf
│   └── Medical Reports.pdf
│
└── tests/
    ├── unit/
    │   ├── core/
    │   │   ├── agents/             # BaseAgent, ScribeAgent, EvaluationAgent tests
    │   │   └── tools/              # SOAPTools, PatientTools, RetrieverTools, etc.
    │   └── infrastructure/
    │       ├── cache/              # CacheService, Redis manager tests
    │       ├── language_models/    # BedrockModel, SonicModel, BaseLLM tests
    │       ├── language_model_service/
    │       └── services/           # SOAPService, EvaluationService, RAG, Patient, Transcription
    └── e2e/
        └── base.py
```

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.11 | Use pyenv or conda to manage versions |
| Docker | 24.x | Required for containerised runs |
| Docker Compose | v2.x (`docker compose`) | Bundled with Docker Desktop |
| GNU Make | 3.81+ | Pre-installed on macOS/Linux |
| AWS account | — | Bedrock + Nova Act access required |

---

## Environment Variables

Copy the template and fill in your values:

```bash
cp .env.example .env
```

### Required variables

| Variable | Description |
|----------|-------------|
| `APP_ENV` | Runtime environment: `development`, `staging`, `production` |
| `LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AWS_ACCESS_KEY` | AWS IAM access key ID with Bedrock permissions |
| `AWS_SECRET_KEY` | AWS IAM secret access key |
| `AWS_REGION_NAME` | AWS region — Sonic requires `us-east-1` |
| `NOVA_ACT_API_KEY` | Nova Act API key for EHR browser automation |
| `SERP_API_KEY` | SERP API key for web search tools |
| `REDIS_HOST` | Redis hostname (`localhost` locally, `redis` in Docker) |
| `REDIS_PORT` | Redis port (default `6379`) |
| `REDIS_DB` | Redis database index (default `0`) |
| `REDIS_PASSWORD` | Redis password |
| `REDIS_URL` | Full Redis URL: `redis://:password@host:port/db` |

### Optional variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_INFERENCE_PROFILE` | `779056097161` | AWS account ID for cross-region inference |
| `REDIS_NAME` | — | Logical name for the Redis instance |
| `REDIS_LOCATION` | — | `local` or `remote` |
| `CHROMA_COLLECTION` | `clinical_docs` | ChromaDB collection name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Path where ChromaDB persists data |
| `API_ROOT_PATH` | `` | Sub-path prefix when behind a reverse proxy |
| `API_PORT` | `8000` | Host port for Docker mappings |

> **Important:** Never commit `.env` or `.docker.env` to version control. Both are listed in `.gitignore`.

---

## Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/medscribe_plus.git
cd medscribe_plus/backend

# 2. Create and activate the virtual environment
make setup
source .venv/bin/activate

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env with your AWS credentials, Redis config, etc.

# 4. Start a local Redis instance (Docker required)
make run-redis-local

# 5. (Optional) Ingest clinical documents into ChromaDB
make ingest
```

---

## Running the API Locally

> Requires Redis to be running (see `make run-redis-local`).

```bash
make run-local
```

The API starts with hot-reload enabled at **http://localhost:8000**.

- Interactive docs: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**
- Health check: **http://localhost:8000/health**

To run on a different port:

```bash
PORT=9000 make run-local
```

---

## Running with Docker

The Docker build validates environment variables before proceeding. If no `.env` or `.docker.env` is found and the required vars are not exported in your shell, the build will fail with a clear error message.

### Build the image

```bash
make build
```

Or with a custom tag:

```bash
IMAGE_TAG=v1.2.3 make build
```

### Run the container (standalone — requires an external Redis)

```bash
make run
```

This command resolves env files in priority order:

1. Variables already exported in your shell
2. `.docker.env` (if present)
3. `.env` (if present)

If none are found, the command aborts with instructions.

---

## Running with Docker Compose (recommended)

Docker Compose starts both the API **and** Redis together. You do not need Redis installed on your machine.

```bash
# Build and start everything
make start

# View API logs
make logs

# View Redis logs
make logs-redis

# Stop everything
make stop

# Stop and remove volumes (wipes Redis data and ChromaDB)
docker compose down -v
```

After `make start`:

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Redis | localhost:6379 |

### How env files are resolved in Docker Compose

The `docker-compose.yml` loads env files in this order (later files take precedence for duplicate keys):

1. `.env`
2. `.docker.env`
3. Shell environment variables (passed via `environment:` block)

The `REDIS_HOST` is always overridden to `redis` (the container service name) so the API can find Redis regardless of what is in your `.env` file.

If you need Docker-specific values (e.g. a remote Redis URL for staging), create `.docker.env`:

```bash
cp .env.example .docker.env
# Edit .docker.env with Docker-specific overrides
```

`.docker.env` is gitignored and takes precedence over `.env`.

---

## Testing

### Run all unit tests

```bash
make test
```

### Run unit tests only

```bash
make test-unit
```

### Run a specific test file

```bash
make test-file FILE=tests/unit/infrastructure/services/test_soap_service.py
```

### Run with coverage report

```bash
make test-cov
# Opens htmlcov/index.html
```

### CI mode (fail fast, no colour)

```bash
bash scripts/run_tests.sh ci
```

### Using the test runner script directly

```bash
bash scripts/run_tests.sh unit          # unit tests
bash scripts/run_tests.sh all           # all tests
bash scripts/run_tests.sh cov           # with coverage
bash scripts/run_tests.sh file tests/unit/core/agents/
bash scripts/run_tests.sh ci            # CI mode
```

### Test structure

Tests live under `tests/unit/` and mirror the `src/` layout:

```
tests/unit/
├── core/
│   ├── agents/       # BaseAgent, ScribeAgent, EvaluationAgent
│   └── tools/        # SOAPTools, PatientTools, RetrieverTools, etc.
└── infrastructure/
    ├── cache/        # CacheService, Redis manager
    ├── language_models/   # BedrockModel, SonicModel
    ├── language_model_service/
    └── services/     # All domain services
```

All tests use **pytest + pytest-mock** exclusively. `unittest.mock` is never imported directly — use `mocker` from the `mocker` fixture.

The root `conftest.py` stubs out heavy SDK modules (`aws_sdk_bedrock_runtime`, `pdf2image`, `pypdf`, `docx2pdf`, PIL, etc.) so tests run without installing those packages or requiring cloud credentials.

---

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make help` | Print all available targets |
| `make setup` | Create virtualenv and install dependencies |
| `make install` | Sync dependencies into existing venv |
| `make run-local` | Start API locally with hot-reload |
| `make run-redis-local` | Start a local dev Redis container |
| `make stop-redis-local` | Stop and remove the local dev Redis container |
| `make test` | Run full test suite |
| `make test-unit` | Run unit tests only |
| `make test-cov` | Run tests with HTML coverage report |
| `make test-watch` | Re-run tests on file change |
| `make test-file FILE=<path>` | Run a single test file or directory |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with ruff |
| `make typecheck` | Run mypy type checker |
| `make build` | Build Docker image |
| `make run` | Run standalone API container |
| `make start` | Build + start full stack (API + Redis) |
| `make stop` | Stop all stack containers |
| `make restart` | Stop then start |
| `make logs` | Tail API container logs |
| `make logs-redis` | Tail Redis container logs |
| `make shell` | Open bash inside running API container |
| `make ps` | Show running containers |
| `make ingest` | Ingest clinical documents into ChromaDB |
| `make clean` | Remove Python artifacts and caches |
| `make clean-docker` | Remove stopped containers and dangling images |
| `make clean-all` | clean + clean-docker + remove venv |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/docs` | OpenAPI documentation |
| `POST` | `/api/v1/scribe/upload` | Upload audio file for transcription + SOAP generation |
| `GET` | `/api/v1/scribe/consultation/{session_id}` | Get consultation result by session |
| `POST` | `/api/v1/scribe/approve` | Approve SOAP note and insert into EHR |
| `WS` | `/api/v1/scribe/ws/{patient_id}` | Real-time audio streaming WebSocket |
| `GET` | `/api/v1/soap/{session_id}` | Retrieve cached SOAP note |
| `POST` | `/api/v1/rag/retrieve` | Semantic search over clinical documents |
| `POST` | `/api/v1/rag/ingest` | Ingest a new clinical document |

Full interactive documentation is available at `/docs` when the server is running.

---

## Ingesting Clinical Documents

Clinical reference PDFs (guidelines, drug references) must be ingested into ChromaDB before the RAG tools can retrieve context for evaluation.

```bash
# Locally
make ingest

# Or directly
python utils/ingest.py
```

Place your PDFs in the `documents/` directory. The ingest script:

1. Parses each PDF into plain text
2. Chunks the text with a sentence-aware sliding window
3. Embeds each chunk via BedrockEmbeddingModel (Titan Embeddings)
4. Upserts all chunks into ChromaDB with metadata

The ChromaDB data is persisted to `./chroma_db/` locally and to a named Docker volume when running via `docker compose`.

---

## Troubleshooting

### `ModuleNotFoundError` during test collection

The root `conftest.py` stubs all heavy SDK modules. If you see a new `ModuleNotFoundError`, add the module to `_MODULES` and its imported names to `_ATTRS` in `conftest.py`. See the comments in that file for the exact pattern.

### `Cannot connect to Redis`

- **Locally**: ensure `make run-redis-local` is running, and that `REDIS_HOST=localhost` in your `.env`.
- **Docker Compose**: the compose file always sets `REDIS_HOST=redis`. If the API container starts before Redis is healthy, `depends_on: condition: service_healthy` will hold it back.

### `AWS credentials not found`

Ensure `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, and `AWS_REGION_NAME` are set in `.env`. Bedrock requires the region to be `us-east-1` for Nova 2 Sonic.

### `ImageError` or `ffmpeg` not found

The Dockerfile installs `ffmpeg` and `poppler-utils` in both the builder and runtime stages. If running locally without Docker, install them manually:

```bash
# macOS
brew install ffmpeg poppler

# Ubuntu / Debian
sudo apt-get install ffmpeg poppler-utils
```

### Docker build fails with `ERROR: No .env or .docker.env found`

Create a `.env` from the template before building:

```bash
cp .env.example .env
# Fill in your values
make build
```

### Port already in use

Change the host port:

```bash
PORT=9000 make run-local
# or
API_PORT=9000 make start
```