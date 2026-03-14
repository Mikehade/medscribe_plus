#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/check_env.sh
#
# Validates that all required environment variables are present before
# allowing a Docker build or run to proceed.
#
# Env resolution order (highest wins):
#   1. Shell environment (already exported vars)
#   2. .docker.env  (Docker-specific overrides)
#   3. .env         (shared dev defaults)
#
# Usage:
#   source scripts/check_env.sh          # exits with code 1 on failure
#   bash scripts/check_env.sh && make build
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# ── Load env files (lower-priority first so higher ones can override) ─────────

load_env_file() {
    local file="$1"
    if [ -f "$file" ]; then
        printf "${YELLOW}  Loading %s${RESET}\n" "$file"
        # Export every non-comment, non-empty line
        set -a
        # shellcheck disable=SC1090
        source <(grep -v '^\s*#' "$file" | grep -v '^\s*$')
        set +a
    fi
}

printf "\n${BOLD}Resolving environment variables...${RESET}\n"
load_env_file ".env"
load_env_file ".docker.env"
printf "  Host shell variables have highest priority.\n\n"

# ── Required variables ────────────────────────────────────────────────────────
# Add or remove variables from this list as the application evolves.

REQUIRED_VARS=(
    "APP_ENV"
    "LOG_LEVEL"
    "AWS_ACCESS_KEY"
    "AWS_SECRET_KEY"
    "AWS_REGION_NAME"
    "NOVA_ACT_API_KEY"
    "SERP_API_KEY"
    "REDIS_HOST"
    "REDIS_PORT"
    "REDIS_DB"
    "REDIS_PASSWORD"
    "REDIS_URL"
)

# ── Optional variables (warn if missing but don't block) ──────────────────────

OPTIONAL_VARS=(
    "AWS_INFERENCE_PROFILE"
    "REDIS_NAME"
    "REDIS_LOCATION"
    "CHROMA_COLLECTION"
    "CHROMA_PERSIST_DIR"
    "API_ROOT_PATH"
)

# ── Validation ────────────────────────────────────────────────────────────────

MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done

# Warn about optional missing vars (non-blocking)
for var in "${OPTIONAL_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        printf "${YELLOW}  WARN: Optional variable %s is not set (using default).${RESET}\n" "$var"
    fi
done

# ── Result ────────────────────────────────────────────────────────────────────

if [ ${#MISSING[@]} -gt 0 ]; then
    printf "\n${RED}${BOLD}ERROR: The following required environment variables are not set:${RESET}\n\n"
    for var in "${MISSING[@]}"; do
        printf "  ${RED}✗ %s${RESET}\n" "$var"
    done
    printf "\n${BOLD}To fix this:${RESET}\n"
    printf "  1. Copy the template:  cp .env.example .env\n"
    printf "  2. Fill in your values in .env\n"
    printf "  3. Or export variables in your shell before running make.\n\n"
    printf "${YELLOW}See .env.example for all variable descriptions.${RESET}\n\n"
    exit 1
fi

printf "${GREEN}${BOLD}✓ All required environment variables are set.${RESET}\n\n"