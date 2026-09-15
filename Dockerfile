FROM ollama/ollama:0.32.15

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt /requirements.txt
RUN uv venv --python 3.11 /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache -r /requirements.txt

# Store the model permanently inside the Docker image
ENV OLLAMA_MODELS=/root/.ollama/models

# Download Tinyrick Q6_K ONCE while building the image
RUN ollama serve > /tmp/ollama.log 2>&1 & \
    OLLAMA_PID=$! && \
    sleep 5 && \
    ollama pull tinyrick/Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved-GGUF:Q6_K && \
    kill $OLLAMA_PID

COPY handler.py /handler.py
COPY test_input.json /test_input.json
COPY start.sh /start.sh

RUN chmod +x /start.sh && \
    sed -i 's|/runpod-volume/ollama/models|/root/.ollama/models|g' /start.sh

ENV OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_CONTEXT_LENGTH=8192

ENTRYPOINT []
CMD ["/start.sh"]
