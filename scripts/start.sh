#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/start.sh
#
# Single entry point to start any combination of services.
# Handles env validation, build, and startup with clear output.
#
# Usage (run from project root medscribe_plus/):
#   bash scripts/start.sh backend      # Redis + API only (default)
#   bash scripts/start.sh full         # Redis + API + Frontend
#   bash scripts/start.sh frontend     # Frontend only (API must already be up)
#   bash scripts/start.sh stop         # Stop everything
#   bash scripts/start.sh logs         # Tail all logs
#   bash scripts/start.sh status       # Show container status
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

MODE="${1:-backend}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# ── Env validation ────────────────────────────────────────────────────────────

check_env() {
    local found=0
    for f in backend/.env backend/.docker.env .env; do
        [ -f "$f" ] && found=1 && break
    done

    if [ $found -eq 0 ]; then
        printf "${RED}${BOLD}ERROR: No .env file found.${RESET}\n\n"
        printf "Create one from the template:\n"
        printf "  ${BOLD}cp backend/env.example backend/.env${RESET}\n"
        printf "  Then fill in your AWS credentials, Redis password, etc.\n\n"
        exit 1
    fi
}

# ── Helpers ───────────────────────────────────────────────────────────────────

wait_for_health() {
    local name="$1"
    local url="$2"
    local max=30
    local i=0
    printf "${YELLOW}Waiting for %s to be healthy...${RESET}" "$name"
    while [ $i -lt $max ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            printf " ${GREEN}${BOLD}ready${RESET}\n"
            return 0
        fi
        printf "."
        sleep 2
        i=$((i + 1))
    done
    printf " ${RED}timed out${RESET}\n"
    return 1
}

# ── Mode handlers ─────────────────────────────────────────────────────────────

start_backend() {
    check_env
    printf "\n${BOLD}Starting backend (Redis + API)...${RESET}\n\n"
    docker compose up --build -d
    wait_for_health "API" "http://localhost:${API_PORT}/health" || {
        printf "\n${RED}API did not become healthy. Check logs:${RESET}\n"
        printf "  docker compose logs api\n\n"
        exit 1
    }
    printf "\n${GREEN}${BOLD}Backend is up!${RESET}\n"
    printf "  API:      http://localhost:${API_PORT}\n"
    printf "  API Docs: http://localhost:${API_PORT}/docs\n\n"
}

start_full() {
    check_env
    printf "\n${BOLD}Starting full stack (Redis + API + Frontend)...${RESET}\n\n"
    docker compose --profile full up --build -d
    wait_for_health "API" "http://localhost:${API_PORT}/health" || {
        printf "\n${RED}API did not become healthy. Check logs:${RESET}\n"
        printf "  docker compose logs api\n\n"
        exit 1
    }
    wait_for_health "Frontend" "http://localhost:${FRONTEND_PORT}/nginx-health" || {
        printf "\n${RED}Frontend did not become healthy. Check logs:${RESET}\n"
        printf "  docker compose --profile full logs frontend\n\n"
        exit 1
    }
    printf "\n${GREEN}${BOLD}Full stack is up!${RESET}\n"
    printf "  Frontend: http://localhost:${FRONTEND_PORT}\n"
    printf "  API:      http://localhost:${API_PORT}\n"
    printf "  API Docs: http://localhost:${API_PORT}/docs\n\n"
}

start_frontend_only() {
    printf "\n${BOLD}Starting frontend only...${RESET}\n\n"
    docker compose --profile full up --build -d frontend
    wait_for_health "Frontend" "http://localhost:${FRONTEND_PORT}/nginx-health"
    printf "\n${GREEN}${BOLD}Frontend is up!${RESET}\n"
    printf "  Frontend: http://localhost:${FRONTEND_PORT}\n\n"
}

stop_all() {
    printf "\n${BOLD}Stopping all services...${RESET}\n"
    docker compose --profile full down
    printf "${GREEN}Done.${RESET}\n\n"
}

show_logs() {
    docker compose --profile full logs -f
}

show_status() {
    printf "\n${BOLD}Container status:${RESET}\n\n"
    docker compose --profile full ps
    printf "\n"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$MODE" in
    backend)  start_backend ;;
    full)     start_full ;;
    frontend) start_frontend_only ;;
    stop)     stop_all ;;
    logs)     show_logs ;;
    status)   show_status ;;
    *)
        printf "${RED}Unknown mode: %s${RESET}\n" "$MODE"
        printf "Valid modes: backend | full | frontend | stop | logs | status\n"
        exit 1
        ;;
esac