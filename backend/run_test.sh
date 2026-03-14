#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_tests.sh
#
# Flexible test runner. Wraps pytest with sensible defaults and lets
# callers choose a mode via the first argument.
#
# Usage:
#   bash scripts/run_tests.sh              # default: full unit suite
#   bash scripts/run_tests.sh unit         # unit tests only
#   bash scripts/run_tests.sh cov          # with HTML coverage report
#   bash scripts/run_tests.sh file <path>  # single file or directory
#   bash scripts/run_tests.sh ci           # CI mode: no colour, exit on first fail
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
RESET="\033[0m"

VENV_DIR=".venv"
PYTEST="$VENV_DIR/bin/pytest"

if [ ! -f "$PYTEST" ]; then
    printf "${RED}${BOLD}ERROR: virtualenv not found at %s${RESET}\n" "$VENV_DIR"
    printf "  Run: ${BOLD}make setup${RESET} first.\n"
    exit 1
fi

MODE="${1:-unit}"

case "$MODE" in

    unit)
        printf "${BOLD}Running unit tests...${RESET}\n\n"
        "$PYTEST" tests/unit/ -v --tb=short
        ;;

    all)
        printf "${BOLD}Running all tests...${RESET}\n\n"
        "$PYTEST" tests/ -v --tb=short
        ;;

    cov)
        printf "${BOLD}Running tests with coverage...${RESET}\n\n"
        "$PYTEST" tests/ \
            --cov=src \
            --cov=utils \
            --cov-report=term-missing \
            --cov-report=html:htmlcov \
            -v
        printf "\n${GREEN}${BOLD}Coverage report written to: htmlcov/index.html${RESET}\n"
        ;;

    file)
        TARGET="${2:-}"
        if [ -z "$TARGET" ]; then
            printf "${RED}ERROR: provide a file or directory path as the second argument.${RESET}\n"
            printf "  Example: bash scripts/run_tests.sh file tests/unit/core/agents/\n"
            exit 1
        fi
        printf "${BOLD}Running tests in: %s${RESET}\n\n" "$TARGET"
        "$PYTEST" "$TARGET" -v --tb=short
        ;;

    ci)
        printf "${BOLD}Running in CI mode (fail-fast, no colour)...${RESET}\n\n"
        "$PYTEST" tests/unit/ \
            --tb=short \
            --no-header \
            -q \
            -x \
            --color=no
        ;;

    *)
        printf "${RED}Unknown mode: %s${RESET}\n" "$MODE"
        printf "Valid modes: unit | all | cov | file <path> | ci\n"
        exit 1
        ;;
esac