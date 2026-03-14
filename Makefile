# ─────────────────────────────────────────────────────────────────────────────
# MedScribe+ — Root Makefile
#
# Manages the full monorepo (backend + frontend) from the project root.
# Also delegates to backend/Makefile for backend-only operations.
#
# Layout:
#   medscribe_plus/
#   ├── docker-compose.yml   ← full stack (this Makefile uses this)
#   ├── Makefile             ← this file
#   ├── backend/
#   │   ├── Makefile         ← backend-only operations
#   │   └── docker-compose.yml
#   └── frontend/
#
# Quick start:
#   make start-backend       start Redis + API only
#   make start               build and start the full stack (API + Frontend)
#   make stop                stop everything
#   make test                run backend unit tests
#   make help                print all targets
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

define check_env
	@if [ ! -f backend/.env ] && [ ! -f backend/.docker.env ] && [ ! -f .env ]; then \
		printf "$(RED)$(BOLD)ERROR:$(RESET) No .env file found.\n"; \
		printf "  Run: $(BOLD)cp backend/env.example backend/.env$(RESET) and fill in your values.\n"; \
		exit 1; \
	fi
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
	@printf "$(BOLD)Building and starting full stack (Redis + API + Frontend)...$(RESET)\n"
	docker compose --profile full up --build -d
	@printf "$(GREEN)$(BOLD)Full stack is up.$(RESET)\n"
	@printf "  Frontend: http://localhost:$(FRONTEND_PORT)\n"
	@printf "  API:      http://localhost:$(API_PORT)\n"
	@printf "  API Docs: http://localhost:$(API_PORT)/docs\n"

.PHONY: stop
stop:
	@printf "$(BOLD)Stopping full stack...$(RESET)\n"
	docker compose --profile full down

.PHONY: restart
restart: stop start

.PHONY: build
build:
	$(call check_env)
	@printf "$(BOLD)Building all Docker images...$(RESET)\n"
	docker compose --profile full build --no-cache

.PHONY: logs
logs:
	docker compose --profile full logs -f

.PHONY: ps
ps:
	docker compose --profile full ps


# ── Backend only ──────────────────────────────────────────────────────────────

.PHONY: start-backend
start-backend:
	$(call check_env)
	@printf "$(BOLD)Starting Redis + API...$(RESET)\n"
	docker compose up --build -d
	@printf "$(GREEN)$(BOLD)Backend is up.$(RESET)\n"
	@printf "  API:      http://localhost:$(API_PORT)\n"
	@printf "  API Docs: http://localhost:$(API_PORT)/docs\n"

.PHONY: stop-backend
stop-backend:
	docker compose down

.PHONY: build-backend
build-backend:
	$(call check_env)
	docker compose build --no-cache api

.PHONY: logs-api
logs-api:
	docker compose logs -f api

.PHONY: logs-redis
logs-redis:
	docker compose logs -f redis

.PHONY: shell-api
shell-api:
	docker exec -it medscribe_api /bin/bash


# ── Frontend only ─────────────────────────────────────────────────────────────

.PHONY: build-frontend
build-frontend:
	docker compose --profile full build --no-cache frontend

.PHONY: logs-frontend
logs-frontend:
	docker compose --profile full logs -f frontend


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
	rm -rf frontend/dist frontend/.vite
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