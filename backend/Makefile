# ─────────────────────────────────────────────────────────────────────────────
# MedScribe+ Makefile
#
# Prerequisites: Python 3.11+, Docker, Docker Compose v2
#
# Quick start:
#   make setup        — create venv + install deps
#   make run-local    — start API locally (requires Redis)
#   make start        — build image + start full stack via docker compose
#   make test         — run the full test suite
#   make help         — print all targets
# ─────────────────────────────────────────────────────────────────────────────

SHELL          := /bin/bash
.DEFAULT_GOAL  := help

# ── Project config ────────────────────────────────────────────────────────────
APP_NAME       := medscribe
IMAGE_NAME     := $(APP_NAME)-api
IMAGE_TAG      ?= latest
CONTAINER_NAME := $(APP_NAME)_api
PYTHON         := python3.11
VENV_DIR       := .venv
VENV_PYTHON    := $(VENV_DIR)/bin/python
VENV_PIP       := $(VENV_DIR)/bin/pip
PORT           ?= 8000

# ── Colour helpers ────────────────────────────────────────────────────────────
BOLD  := \033[1m
GREEN := \033[32m
YELLOW:= \033[33m
RED   := \033[31m
RESET := \033[0m

# ── Env file resolution ───────────────────────────────────────────────────────
# Priority: .docker.env > .env > host environment
# For local targets we only need .env (or host vars).
# Docker targets pass both files; the container's env_file directive handles
# precedence on the Docker side.

define check_env
	@if [ ! -f .env ] && [ ! -f .docker.env ]; then \
		printf "$(RED)$(BOLD)ERROR:$(RESET) No .env or .docker.env found.\n"; \
		printf "  Run: $(BOLD)cp .env.example .env$(RESET) and fill in your values.\n"; \
		exit 1; \
	fi
endef

define load_env
	$(if $(wildcard .env),export $$(grep -v '^#' .env | grep -v '^$$' | xargs),)
endef

# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "\n$(BOLD)MedScribe+ — available targets$(RESET)\n\n"
	@printf "$(BOLD)$(GREEN)Local development$(RESET)\n"
	@printf "  $(BOLD)setup$(RESET)              Create virtualenv and install all dependencies\n"
	@printf "  $(BOLD)install$(RESET)            Install / sync dependencies into existing venv\n"
	@printf "  $(BOLD)run-local$(RESET)          Start the API with uvicorn (hot-reload, no Docker)\n"
	@printf "  $(BOLD)run-redis-local$(RESET)    Start a Redis container for local development only\n"
	@printf "  $(BOLD)stop-redis-local$(RESET)   Stop and remove the local dev Redis container\n"
	@printf "\n$(BOLD)$(GREEN)Testing$(RESET)\n"
	@printf "  $(BOLD)test$(RESET)               Run the full unit test suite\n"
	@printf "  $(BOLD)test-unit$(RESET)          Run unit tests only\n"
	@printf "  $(BOLD)test-cov$(RESET)           Run tests with HTML coverage report\n"
	@printf "  $(BOLD)test-watch$(RESET)         Re-run tests on file changes (requires pytest-watch)\n"
	@printf "  $(BOLD)lint$(RESET)               Run ruff linter\n"
	@printf "  $(BOLD)format$(RESET)             Auto-format with ruff\n"
	@printf "  $(BOLD)typecheck$(RESET)          Run mypy type checker\n"
	@printf "\n$(BOLD)$(GREEN)Docker$(RESET)\n"
	@printf "  $(BOLD)build$(RESET)              Build the Docker image\n"
	@printf "  $(BOLD)run$(RESET)                Run the API container (requires running Redis)\n"
	@printf "  $(BOLD)start$(RESET)              Build image + start full stack (API + Redis)\n"
	@printf "  $(BOLD)stop$(RESET)               Stop and remove all stack containers\n"
	@printf "  $(BOLD)restart$(RESET)            Stop then start the full stack\n"
	@printf "  $(BOLD)logs$(RESET)               Tail API container logs\n"
	@printf "  $(BOLD)logs-redis$(RESET)         Tail Redis container logs\n"
	@printf "  $(BOLD)shell$(RESET)              Open a bash shell inside the running API container\n"
	@printf "  $(BOLD)ps$(RESET)                 Show running containers\n"
	@printf "\n$(BOLD)$(GREEN)Maintenance$(RESET)\n"
	@printf "  $(BOLD)clean$(RESET)              Remove __pycache__, .pyc files, pytest cache\n"
	@printf "  $(BOLD)clean-docker$(RESET)       Remove stopped containers and dangling images\n"
	@printf "  $(BOLD)clean-all$(RESET)          clean + clean-docker + remove venv\n"
	@printf "  $(BOLD)ingest$(RESET)             Ingest clinical documents into ChromaDB\n"
	@printf "\n"


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: setup
setup:
	@printf "$(BOLD)Creating virtualenv at $(VENV_DIR)...$(RESET)\n"
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@printf "$(GREEN)$(BOLD)Done.$(RESET) Activate with: source $(VENV_DIR)/bin/activate\n"

.PHONY: install
install:
	@printf "$(BOLD)Installing / syncing dependencies...$(RESET)\n"
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

.PHONY: run-local
run-local:
	$(call check_env)
	@printf "$(BOLD)Starting API locally on port $(PORT)...$(RESET)\n"
	@set -a && source .env 2>/dev/null || true && set +a && \
	$(VENV_DIR)/bin/uvicorn main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level info

.PHONY: run-redis-local
run-redis-local:
	@printf "$(BOLD)Starting local dev Redis on port 6379...$(RESET)\n"
	@REDIS_PASSWORD=$$(grep REDIS_PASSWORD .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo "changeme"); \
	docker run -d \
		--name medscribe_redis_dev \
		-p 6379:6379 \
		-e REDIS_PASSWORD=$$REDIS_PASSWORD \
		redis:7.2-alpine \
		redis-server --requirepass $$REDIS_PASSWORD || \
	printf "$(YELLOW)Redis container already running or port in use.$(RESET)\n"

.PHONY: stop-redis-local
stop-redis-local:
	@printf "$(BOLD)Stopping local dev Redis...$(RESET)\n"
	docker stop medscribe_redis_dev 2>/dev/null || true
	docker rm medscribe_redis_dev 2>/dev/null || true


# ─────────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	@printf "$(BOLD)Running full test suite...$(RESET)\n"
	$(VENV_DIR)/bin/pytest tests/ -v --tb=short

.PHONY: test-unit
test-unit:
	@printf "$(BOLD)Running unit tests...$(RESET)\n"
	$(VENV_DIR)/bin/pytest tests/unit/ -v --tb=short

.PHONY: test-cov
test-cov:
	@printf "$(BOLD)Running tests with coverage...$(RESET)\n"
	$(VENV_DIR)/bin/pytest tests/ \
		--cov=src \
		--cov=utils \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v
	@printf "$(GREEN)Coverage report: $(BOLD)htmlcov/index.html$(RESET)\n"

.PHONY: test-watch
test-watch:
	@printf "$(BOLD)Watching for changes...$(RESET)\n"
	$(VENV_DIR)/bin/ptw tests/unit/ -- -v --tb=short

.PHONY: test-file
test-file:
	@printf "$(BOLD)Running tests in $(FILE)...$(RESET)\n"
	$(VENV_DIR)/bin/pytest $(FILE) -v --tb=short

.PHONY: lint
lint:
	@printf "$(BOLD)Linting with ruff...$(RESET)\n"
	$(VENV_DIR)/bin/ruff check src/ utils/ tests/

.PHONY: format
format:
	@printf "$(BOLD)Formatting with ruff...$(RESET)\n"
	$(VENV_DIR)/bin/ruff format src/ utils/ tests/
	$(VENV_DIR)/bin/ruff check --fix src/ utils/ tests/

.PHONY: typecheck
typecheck:
	@printf "$(BOLD)Running mypy...$(RESET)\n"
	$(VENV_DIR)/bin/mypy src/ --ignore-missing-imports


# ─────────────────────────────────────────────────────────────────────────────
# DOCKER
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build
build:
	$(call check_env)
	@printf "$(BOLD)Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)\n"
	docker build \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		.
	@printf "$(GREEN)$(BOLD)Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(RESET)\n"

.PHONY: run
run:
	$(call check_env)
	@printf "$(BOLD)Running $(IMAGE_NAME):$(IMAGE_TAG) on port $(PORT)...$(RESET)\n"
	@ENV_FILE_ARGS=""; \
	if [ -f .docker.env ]; then ENV_FILE_ARGS="$$ENV_FILE_ARGS --env-file .docker.env"; fi; \
	if [ -f .env ];        then ENV_FILE_ARGS="$$ENV_FILE_ARGS --env-file .env"; fi; \
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		$$ENV_FILE_ARGS \
		-p $(PORT):8000 \
		-v $$(pwd)/chroma_db:/app/chroma_db \
		-v $$(pwd)/temp_uploads:/app/temp_uploads \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: start
start:
	$(call check_env)
	@printf "$(BOLD)Building and starting full stack (API + Redis)...$(RESET)\n"
	docker compose up --build -d
	@printf "$(GREEN)$(BOLD)Stack is up.$(RESET)\n"
	@printf "  API:   http://localhost:$(PORT)\n"
	@printf "  Docs:  http://localhost:$(PORT)/docs\n"
	@printf "  Redis: localhost:6379\n"

.PHONY: stop
stop:
	@printf "$(BOLD)Stopping stack...$(RESET)\n"
	docker compose down

.PHONY: restart
restart: stop start

.PHONY: logs
logs:
	docker compose logs -f api

.PHONY: logs-redis
logs-redis:
	docker compose logs -f redis

.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

.PHONY: ps
ps:
	docker compose ps


# ─────────────────────────────────────────────────────────────────────────────
# MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: ingest
ingest:
	@printf "$(BOLD)Ingesting clinical documents into ChromaDB...$(RESET)\n"
	@set -a && source .env 2>/dev/null || true && set +a && \
	$(VENV_PYTHON) utils/ingest.py

.PHONY: clean
clean:
	@printf "$(BOLD)Cleaning Python artifacts...$(RESET)\n"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@printf "$(GREEN)Clean.$(RESET)\n"

.PHONY: clean-docker
clean-docker:
	@printf "$(BOLD)Removing stopped containers and dangling images...$(RESET)\n"
	docker container prune -f
	docker image prune -f

.PHONY: clean-all
clean-all: clean clean-docker
	@printf "$(BOLD)Removing virtualenv...$(RESET)\n"
	rm -rf $(VENV_DIR)
	@printf "$(GREEN)All clean.$(RESET)\n"