#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}Stopping Ark Chatbot...${NC}"

# Stop uvicorn
if [ -f .pid ]; then
    PID=$(cat .pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Stopped server (PID $PID)"
    fi
    rm -f .pid
fi

# Kill any remaining uvicorn processes for this project
pkill -f "uvicorn app.main:app" 2>/dev/null && echo "Cleaned up remaining processes" || true

# Stop PostgreSQL
docker compose down 2>/dev/null && echo "Stopped PostgreSQL" || true

echo -e "${RED}Stopped.${NC}"
