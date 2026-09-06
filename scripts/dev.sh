#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SERVICE_NAME="mcxforge"
COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE="${DOCKER_COMPOSE:-}"

if [ -z "$DOCKER_COMPOSE" ]; then
    if command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE="$(command -v docker-compose)"
    elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE="docker compose"
    elif [ -x /usr/local/bin/docker-compose ]; then
        DOCKER_COMPOSE="/usr/local/bin/docker-compose"
    elif [ -x /opt/homebrew/bin/docker-compose ]; then
        DOCKER_COMPOSE="/opt/homebrew/bin/docker-compose"
    else
        echo "❌ docker-compose not found. PATH=$PATH"
        exit 127
    fi
fi

echo "👉 Running command: $1"

if [ "$1" == "b" ]; then
    echo "🔨 Building containers..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE build

elif [ "$1" == "rb" ]; then
    echo "♻️ Rebuilding WITHOUT cache..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE build --no-cache

elif [ "$1" == "up" ]; then
    echo "🚀 Starting containers..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE up -d

elif [ "$1" == "r" ]; then
    echo "🔁 Restarting containers..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE restart

elif [ "$1" == "fr" ]; then
    echo "♻️ Force recreating containers with fresh .env and code..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE up -d --force-recreate

elif [ "$1" == "down" ]; then
    echo "🛑 Stopping containers..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE down

elif [ "$1" == "clean" ]; then
    echo "🔥 FULL CLEAN (containers + images)..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE down --rmi all --volumes --remove-orphans

elif [ "$1" == "logs" ]; then
    echo "📜 Logs..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f

elif [ "$1" == "shell" ]; then
    echo "🐚 Entering container shell..."
    docker exec -it $SERVICE_NAME sh

else
    echo "❌ Unknown command"
    echo "Usage:"
    echo "  ./dev.sh b      → build"
    echo "  ./dev.sh rb     → rebuild (no cache)"
    echo "  ./dev.sh up     → start"
    echo "  ./dev.sh r      → restart"
    echo "  ./dev.sh fr     → force recreate (fresh .env & code)"
    echo "  ./dev.sh down   → stop"
    echo "  ./dev.sh clean  → full reset"
    echo "  ./dev.sh logs   → logs"
    echo "  ./dev.sh shell  → container shell"
fi
