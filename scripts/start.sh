#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/start.sh — run from project root (medscribe_plus/)
#
# Usage:
#   bash scripts/start.sh backend      # Redis + API only (default)
#   bash scripts/start.sh full         # Redis + API + Frontend
#   bash scripts/start.sh frontend     # Frontend only
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

# ── Load env into shell so compose ${VAR} substitutions use real values ───────
load_env() {
    for f in backend/.env backend/.docker.env .env; do
        if [ -f "$f" ]; then
            printf "${YELLOW}  Exporting env from: %s${RESET}\n" "$f"
            set -a
            # shellcheck disable=SC1090
            source "$f"
            set +a
            return 0
        fi
    done
    printf "${RED}${BOLD}ERROR: No .env file found.${RESET}\n"
    printf "  Run: cp backend/env.example backend/.env\n\n"
    exit 1
}

# ── Write frontend/.env.production and patch hardcoded URLs ──────────────────
setup_frontend_env() {
    printf "${YELLOW}  Writing frontend/.env.production...${RESET}\n"
    cat > frontend/.env.production << 'EOF'
VITE_API_URL=/api/v1
VITE_WS_URL=__DYNAMIC__
EOF

    # Patch LiveRecording.tsx to use dynamic WS URL based on window.location.host
    # so it works in Codespaces, local, and production without hardcoded localhost
    if grep -q "localhost:8000" frontend/src/pages/LiveRecording.tsx 2>/dev/null; then
        printf "${YELLOW}  Patching LiveRecording.tsx — replacing hardcoded WS URL...${RESET}\n"
        sed -i 's|import\.meta\.env\.VITE_WS_URL || "ws://localhost:8000/api/transcribe/"|`${window.location.protocol === '"'"'https:'"'"' ? '"'"'wss:'"'"' : '"'"'ws:'"'"'}://${window.location.host}/api/transcribe/`|g' \
            frontend/src/pages/LiveRecording.tsx
    fi

    # Patch the approve URL that also has hardcoded localhost
    if grep -q "localhost:8020" frontend/src/pages/LiveRecording.tsx 2>/dev/null; then
        printf "${YELLOW}  Patching LiveRecording.tsx — replacing hardcoded approve URL...${RESET}\n"
        sed -i 's|http://localhost:8020||g' frontend/src/pages/LiveRecording.tsx
    fi

    printf "${GREEN}  Frontend env ready.${RESET}\n"
}

# ── Health check helper ───────────────────────────────────────────────────────
wait_for_health() {
    local name="$1"
    local url="$2"
    local max=30
    local i=0
    printf "${YELLOW}  Waiting for %s...${RESET}" "$name"
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
    load_env
    printf "\n${BOLD}Starting backend (Redis + API)...${RESET}\n\n"
    docker compose up --build -d
    wait_for_health "API" "http://localhost:${API_PORT}/health" || {
        printf "\n${RED}API unhealthy. Check logs:${RESET} docker compose logs api\n\n"
        exit 1
    }
    printf "\n${GREEN}${BOLD}Backend is up!${RESET}\n"
    printf "  API:      http://localhost:${API_PORT}\n"
    printf "  API Docs: http://localhost:${API_PORT}/docs\n\n"
}

start_full() {
    load_env
    setup_frontend_env
    printf "\n${BOLD}Starting full stack (Redis + API + Frontend)...${RESET}\n\n"
    docker compose --profile full up --build -d
    wait_for_health "API" "http://localhost:${API_PORT}/health" || {
        printf "\n${RED}API unhealthy. Check logs:${RESET} docker compose logs api\n\n"
        exit 1
    }
    wait_for_health "Frontend" "http://localhost:${FRONTEND_PORT}/nginx-health" || {
        printf "\n${RED}Frontend unhealthy. Check logs:${RESET} docker compose --profile full logs frontend\n\n"
        exit 1
    }
    printf "\n${GREEN}${BOLD}Full stack is up!${RESET}\n"
    printf "  Frontend: http://localhost:${FRONTEND_PORT}\n"
    printf "  API:      http://localhost:${API_PORT}\n"
    printf "  API Docs: http://localhost:${API_PORT}/docs\n\n"
}

start_frontend_only() {
    load_env
    setup_frontend_env
    printf "\n${BOLD}Starting frontend only...${RESET}\n\n"
    docker compose --profile full up --build -d frontend
    wait_for_health "Frontend" "http://localhost:${FRONTEND_PORT}/nginx-health"
    printf "\n${GREEN}${BOLD}Frontend is up!${RESET}\n"
    printf "  Frontend: http://localhost:${FRONTEND_PORT}\n\n"
}

stop_all() {
    load_env
    printf "\n${BOLD}Stopping all services...${RESET}\n"
    docker compose --profile full down
    printf "${GREEN}Done.${RESET}\n\n"
}

show_logs() {
    load_env
    docker compose --profile full logs -f
}

show_status() {
    load_env
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