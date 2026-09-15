FROM ollama/ollama:0.32.15

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt /requirements.txt
RUN uv venv --python 3.11 /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache -r /requirements.txt

COPY handler.py /handler.py
COPY test_input.json /test_input.json
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_CONTEXT_LENGTH=8192

ENTRYPOINT []
CMD ["/start.sh"]
