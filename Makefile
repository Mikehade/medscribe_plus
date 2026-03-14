# ─────────────────────────────────────────────────────────────────────────────
# MedScribe+ — Root Makefile
# ─────────────────────────────────────────────────────────────────────────────

SHELL         := /bin/bash
.DEFAULT_GOAL := help

BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

API_PORT      ?= 8000
FRONTEND_PORT ?= 3000

# ── Env file resolution ───────────────────────────────────────────────────────
ENV_FILE := $(shell \
  if [ -f backend/.env ]; then echo backend/.env; \
  elif [ -f backend/.docker.env ]; then echo backend/.docker.env; \
  elif [ -f .env ]; then echo .env; \
  fi)

load_env = set -a && source $(ENV_FILE) && set +a

define check_env
	@if [ -z "$(ENV_FILE)" ]; then \
		printf "$(RED)$(BOLD)ERROR:$(RESET) No .env file found.\n"; \
		printf "  Run: $(BOLD)cp backend/env.example backend/.env$(RESET) and fill in your values.\n"; \
		exit 1; \
	fi
	@printf "$(YELLOW)  Loading env from: $(ENV_FILE)$(RESET)\n"
endef

# ── Frontend env setup ────────────────────────────────────────────────────────
# Writes frontend/.env.production before every Docker build so Vite bakes
# relative URLs into the bundle instead of hardcoded localhost:8000.
# The WebSocket URL is set to a placeholder — LiveRecording.tsx builds the
# real WS URL dynamically from window.location.host at runtime.
#
# Also patches LiveRecording.tsx to use dynamic WS URL if not already patched.

define setup_frontend_env
	@printf "$(YELLOW)  Writing frontend/.env.production...$(RESET)\n"
	@printf "VITE_API_URL=/api/v1\nVITE_WS_URL=__DYNAMIC__\n" > frontend/.env.production
	@if grep -q "localhost:8000" frontend/src/pages/LiveRecording.tsx 2>/dev/null; then \
		printf "$(YELLOW)  Patching LiveRecording.tsx WS URL to use window.location.host...$(RESET)\n"; \
		sed -i 's|import.meta.env.VITE_WS_URL || "ws://localhost:8000/api/transcribe/"|`$${window.location.protocol === '"'"'https:'"'"' ? '"'"'wss:'"'"' : '"'"'ws:'"'"'}://$${window.location.host}/api/transcribe/`|g' \
			frontend/src/pages/LiveRecording.tsx; \
	fi
	@if grep -q "localhost:8020" frontend/src/pages/LiveRecording.tsx 2>/dev/null; then \
		printf "$(YELLOW)  Patching LiveRecording.tsx approve URL...$(RESET)\n"; \
		sed -i 's|http://localhost:8020|""|g' frontend/src/pages/LiveRecording.tsx; \
	fi
	@printf "$(GREEN)  Frontend env ready.$(RESET)\n"
endef

# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "\n$(BOLD)MedScribe+ — available targets$(RESET)\n\n"
	@printf "$(BOLD)$(GREEN)Full stack (backend + frontend)$(RESET)\n"
	@printf "  $(BOLD)start$(RESET)              Build and start the full stack (API + Frontend + Redis)\n"
	@printf "  $(BOLD)stop$(RESET)               Stop the full stack\n"
	@printf "  $(BOLD)restart$(RESET)            Stop then start the full stack\n"
	@printf "  $(BOLD)build$(RESET)              Build all Docker images without starting\n"
	@printf "  $(BOLD)logs$(RESET)               Tail all container logs\n"
	@printf "  $(BOLD)ps$(RESET)                 Show running containers\n"
	@printf "\n$(BOLD)$(GREEN)Backend only$(RESET)\n"
	@printf "  $(BOLD)start-backend$(RESET)      Build and start Redis + API only\n"
	@printf "  $(BOLD)stop-backend$(RESET)       Stop backend containers\n"
	@printf "  $(BOLD)build-backend$(RESET)      Build the backend Docker image only\n"
	@printf "  $(BOLD)logs-api$(RESET)           Tail API container logs\n"
	@printf "  $(BOLD)logs-redis$(RESET)         Tail Redis container logs\n"
	@printf "  $(BOLD)shell-api$(RESET)          Open bash inside the running API container\n"
	@printf "\n$(BOLD)$(GREEN)Frontend only$(RESET)\n"
	@printf "  $(BOLD)build-frontend$(RESET)     Build the frontend Docker image only\n"
	@printf "  $(BOLD)logs-frontend$(RESET)      Tail frontend container logs\n"
	@printf "\n$(BOLD)$(GREEN)Local development (no Docker)$(RESET)\n"
	@printf "  $(BOLD)setup-backend$(RESET)      Create backend venv and install Python deps\n"
	@printf "  $(BOLD)setup-frontend$(RESET)     Install frontend npm deps\n"
	@printf "  $(BOLD)run-backend-local$(RESET)  Start backend API locally with hot-reload\n"
	@printf "  $(BOLD)run-frontend-local$(RESET) Start frontend Vite dev server\n"
	@printf "  $(BOLD)run-redis-local$(RESET)    Start a local Redis container for dev\n"
	@printf "\n$(BOLD)$(GREEN)Testing$(RESET)\n"
	@printf "  $(BOLD)test$(RESET)               Run backend unit tests\n"
	@printf "  $(BOLD)test-cov$(RESET)           Run backend tests with coverage report\n"
	@printf "\n$(BOLD)$(GREEN)Maintenance$(RESET)\n"
	@printf "  $(BOLD)clean$(RESET)              Remove build artifacts and caches\n"
	@printf "  $(BOLD)clean-docker$(RESET)       Remove stopped containers and dangling images\n"
	@printf "  $(BOLD)clean-all$(RESET)          Full clean including node_modules and venv\n"
	@printf "  $(BOLD)ingest$(RESET)             Ingest clinical documents into ChromaDB\n"
	@printf "\n"


# ── Full stack ────────────────────────────────────────────────────────────────

.PHONY: start
start:
	$(call check_env)
	$(call setup_frontend_env)
	@printf "$(BOLD)Building and starting full stack (Redis + API + Frontend)...$(RESET)\n"
	@$(load_env) && docker compose --profile full up --build -d
	@printf "$(GREEN)$(BOLD)Full stack is up.$(RESET)\n"
	@printf "  Frontend: http://localhost:$(FRONTEND_PORT)\n"
	@printf "  API:      http://localhost:$(API_PORT)\n"
	@printf "  API Docs: http://localhost:$(API_PORT)/docs\n"

.PHONY: stop
stop:
	@printf "$(BOLD)Stopping full stack...$(RESET)\n"
	@$(load_env) && docker compose --profile full down

.PHONY: restart
restart: stop start

.PHONY: build
build:
	$(call check_env)
	$(call setup_frontend_env)
	@printf "$(BOLD)Building all Docker images...$(RESET)\n"
	@$(load_env) && docker compose --profile full build --no-cache

.PHONY: logs
logs:
	@$(load_env) && docker compose --profile full logs -f

.PHONY: ps
ps:
	@$(load_env) && docker compose --profile full ps


# ── Backend only ──────────────────────────────────────────────────────────────

.PHONY: start-backend
start-backend:
	$(call check_env)
	@printf "$(BOLD)Starting Redis + API...$(RESET)\n"
	@$(load_env) && docker compose up --build -d
	@printf "$(GREEN)$(BOLD)Backend is up.$(RESET)\n"
	@printf "  API:      http://localhost:$(API_PORT)\n"
	@printf "  API Docs: http://localhost:$(API_PORT)/docs\n"

.PHONY: stop-backend
stop-backend:
	@$(load_env) && docker compose down

.PHONY: build-backend
build-backend:
	$(call check_env)
	@$(load_env) && docker compose build --no-cache api

.PHONY: logs-api
logs-api:
	@$(load_env) && docker compose logs -f api

.PHONY: logs-redis
logs-redis:
	@$(load_env) && docker compose logs -f redis

.PHONY: shell-api
shell-api:
	docker exec -it medscribe_api /bin/bash


# ── Frontend only ─────────────────────────────────────────────────────────────

.PHONY: build-frontend
build-frontend:
	$(call check_env)
	$(call setup_frontend_env)
	@$(load_env) && docker compose --profile full build --no-cache frontend

.PHONY: logs-frontend
logs-frontend:
	@$(load_env) && docker compose --profile full logs -f frontend


# ── Local development ─────────────────────────────────────────────────────────

.PHONY: setup-backend
setup-backend:
	$(MAKE) -C backend setup

.PHONY: setup-frontend
setup-frontend:
	@printf "$(BOLD)Installing frontend npm dependencies...$(RESET)\n"
	cd frontend && npm install

.PHONY: run-backend-local
run-backend-local:
	$(MAKE) -C backend run-local

.PHONY: run-frontend-local
run-frontend-local:
	@printf "$(BOLD)Starting Vite dev server on port $(FRONTEND_PORT)...$(RESET)\n"
	cd frontend && npm run dev -- --port $(FRONTEND_PORT)

.PHONY: run-redis-local
run-redis-local:
	$(MAKE) -C backend run-redis-local


# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(MAKE) -C backend test

.PHONY: test-cov
test-cov:
	$(MAKE) -C backend test-cov


# ── Maintenance ───────────────────────────────────────────────────────────────

.PHONY: ingest
ingest:
	$(MAKE) -C backend ingest

.PHONY: clean
clean:
	$(MAKE) -C backend clean
	@printf "$(BOLD)Cleaning frontend build artifacts...$(RESET)\n"
	rm -rf frontend/dist frontend/.vite frontend/.env.production
	@printf "$(GREEN)Clean.$(RESET)\n"

.PHONY: clean-docker
clean-docker:
	docker container prune -f
	docker image prune -f

.PHONY: clean-all
clean-all: clean clean-docker
	$(MAKE) -C backend clean-all
	@printf "$(BOLD)Removing node_modules...$(RESET)\n"
	rm -rf frontend/node_modules