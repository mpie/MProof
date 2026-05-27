#!/bin/bash

# MProof Startup Script

set -e

echo "Starting MProof..."
echo

# ── Python / venv ─────────────────────────────────────────────────────────────
BACKEND_PYTHON="python3"
if command -v pyenv >/dev/null 2>&1; then
    PYENV_PREFIX="$(pyenv prefix 3.11.8 2>/dev/null || true)"
    [ -x "$PYENV_PREFIX/bin/python" ] && BACKEND_PYTHON="$PYENV_PREFIX/bin/python"
fi
BACKEND_PYTHON_VERSION="$($BACKEND_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python: $($BACKEND_PYTHON --version)"

if [ -d "backend/venv" ]; then
    VENV_VER="$(backend/venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo unknown)"
    if [ "$VENV_VER" != "$BACKEND_PYTHON_VERSION" ]; then
        echo "venv Python mismatch ($VENV_VER vs $BACKEND_PYTHON_VERSION) — recreating..."
        rm -rf backend/venv
    fi
fi

cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$BACKEND_PYTHON" -m venv venv
fi

source venv/bin/activate

# Install / update dependencies
echo "Checking dependencies..."
pip install -r requirements.txt -q

# Verify MySQL driver
if ! python -c "import aiomysql" >/dev/null 2>&1; then
    echo "Installing aiomysql..."
    pip install aiomysql==0.2.0 -q
fi

# ── Database ──────────────────────────────────────────────────────────────────
echo "Running database migrations..."
alembic upgrade head
echo "✓ Database up to date"

mkdir -p data

echo "✓ Backend setup complete"
echo
cd ..

# ── Port checks ───────────────────────────────────────────────────────────────
for PORT in 8000 3000; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $PORT is already in use."
        read -p "Kill existing process on :$PORT? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
            sleep 1
            echo "✓ Killed"
        else
            echo "Exiting — kill port $PORT manually and retry."
            exit 1
        fi
    fi
done

# ── Start backend ─────────────────────────────────────────────────────────────
echo "Starting backend..."
cd backend
source venv/bin/activate
venv/bin/uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "✓ Backend: http://localhost:8000 (PID: $BACKEND_PID)"
cd ..

sleep 3

# ── Start frontend ────────────────────────────────────────────────────────────
echo "Starting frontend..."
cd frontend
export PATH="/opt/homebrew/opt/node@20/bin:$PATH"
[ ! -d "node_modules" ] && npm install -q
npm run dev &
FRONTEND_PID=$!
echo "✓ Frontend: http://localhost:3000 (Node $(node --version))"
cd ..

echo
echo "System started."
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop"

cleanup() {
    echo
    echo "Stopping..."
    kill -TERM $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    for i in {1..5}; do
        kill -0 $BACKEND_PID 2>/dev/null || kill -0 $FRONTEND_PID 2>/dev/null || { echo "✓ Stopped"; exit 0; }
        sleep 1
    done
    kill -9 $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit
}
trap cleanup INT TERM
wait
