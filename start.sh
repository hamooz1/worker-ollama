#!/bin/bash
set -e

echo "Starting Runpod Ollama worker..."

# Cache models on the network volume when one is attached, so warm workers
# and restarts skip the pull entirely.
if [ -d "/runpod-volume" ]; then
    export OLLAMA_MODELS="/runpod-volume/ollama/models"
    mkdir -p "$OLLAMA_MODELS"
    echo "Network volume detected — caching models at $OLLAMA_MODELS"
fi

ollama serve &

echo "Waiting for Ollama server to be ready..."
until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do
    sleep 0.5
done
echo "Ollama server is ready"

if [ -n "$OLLAMA_MODEL" ]; then
    echo "Pulling model: $OLLAMA_MODEL"
    # ensure_model uses HF_TOKEN for gated hf.co models, ollama pull otherwise
    if /opt/venv/bin/python -c 'import os, handler; handler.ensure_model(os.environ["OLLAMA_MODEL"])'; then
        echo "Model pulled: $OLLAMA_MODEL"
    else
        echo "WARN: startup pull of $OLLAMA_MODEL failed — handler will retry on first request"
    fi
fi

cd /
exec /opt/venv/bin/python -u /handler.py
