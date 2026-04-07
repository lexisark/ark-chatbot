#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Ark Chatbot...${NC}"

# Start PostgreSQL if not running
if ! docker compose ps db --status running --format json 2>/dev/null | grep -q "running"; then
    echo -e "${YELLOW}Starting PostgreSQL...${NC}"
    docker compose up -d db
    sleep 3
else
    echo "PostgreSQL already running"
fi

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
fi

# Create database if needed
PGPASSWORD=postgres psql -h localhost -U postgres -lqt 2>/dev/null | grep -qw ark_chatbot || {
    echo -e "${YELLOW}Creating database...${NC}"
    PGPASSWORD=postgres psql -h localhost -U postgres -c "CREATE DATABASE ark_chatbot;" 2>/dev/null
    PGPASSWORD=postgres psql -h localhost -U postgres -d ark_chatbot -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;" 2>/dev/null
}

# Start app
PORT=${PORT:-8000}
echo -e "${GREEN}Starting server on port ${PORT}...${NC}"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload &
echo $! > .pid

echo -e "${GREEN}Ark Chatbot running at http://localhost:${PORT}${NC}"
