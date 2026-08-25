import hashlib
import json
import os

import requests
import runpod

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS", "/root/.ollama/models")

session = requests.Session()


def get_local_models():
    response = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def pull_hf_model_with_token(model):
    """Pull an hf.co model using an HF access token (gated/private repos).

    Ollama's native pull authenticates with an SSH key, which serverless
    workers can't practically register. Instead, fetch the same manifest and
    blobs from Hugging Face's Ollama-compatible registry with bearer auth and
    write them straight into Ollama's model store.
    """
    ref = model.split("/", 1)[1]  # strip "hf.co/" or "huggingface.co/"
    repo, _, tag = ref.partition(":")
    tag = tag or "latest"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    manifest_response = session.get(
        f"https://huggingface.co/v2/{repo}/manifests/{tag}",
        headers={**headers, "Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        timeout=60,
    )
    manifest_response.raise_for_status()
    manifest = manifest_response.json()

    blobs_dir = os.path.join(OLLAMA_MODELS_DIR, "blobs")
    manifest_dir = os.path.join(OLLAMA_MODELS_DIR, "manifests", "hf.co", repo)
    os.makedirs(blobs_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    for layer in [manifest["config"], *manifest["layers"]]:
        digest = layer["digest"]
        blob_path = os.path.join(blobs_dir, digest.replace(":", "-"))
        if os.path.exists(blob_path) and os.path.getsize(blob_path) == layer["size"]:
            continue
        hasher = hashlib.sha256()
        with session.get(
            f"https://huggingface.co/v2/{repo}/blobs/{digest}",
            headers=headers,
            stream=True,
            timeout=3600,
        ) as blob_response:
            blob_response.raise_for_status()
            with open(blob_path + ".partial", "wb") as f:
                for chunk in blob_response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    hasher.update(chunk)
        if f"sha256:{hasher.hexdigest()}" != digest:
            os.remove(blob_path + ".partial")
            raise ValueError(f"Digest mismatch downloading blob {digest} for {model}")
        os.replace(blob_path + ".partial", blob_path)

    with open(os.path.join(manifest_dir, tag), "w") as f:
        json.dump(manifest, f)


def ensure_model(model):
    local = get_local_models()
    if model in local or f"{model}:latest" in local:
        return
    if HF_TOKEN and model.startswith(("hf.co/", "huggingface.co/")):
        pull_hf_model_with_token(model)
        return
    response = session.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model, "stream": False},
        timeout=3600,
    )
    response.raise_for_status()


def handler(job):
    job_input = job.get("input") or {}

    model = job_input.get("model") or DEFAULT_MODEL
    if not model:
        yield {
            "error": (
                "No model specified. Set the OLLAMA_MODEL environment variable "
                "on the endpoint or pass 'model' in the request input, e.g. "
                "'hf.co/prism-ml/Bonsai-27B-gguf:Q1_0' or 'llama3.2:3b'."
            )
        }
        return

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if not messages and not prompt:
        yield {"error": "Provide either 'messages' (chat) or 'prompt' (completion) in input."}
        return

    try:
        ensure_model(model)
    except (requests.RequestException, ValueError) as err:
        yield {"error": f"Failed to pull model '{model}': {err}"}
        return

    if messages:
        endpoint = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {"model": model, "messages": messages}
    else:
        endpoint = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {"model": model, "prompt": prompt}

    for key in ("options", "format", "keep_alive", "tools", "system", "template"):
        if key in job_input:
            payload[key] = job_input[key]

    stream = bool(job_input.get("stream", False))
    payload["stream"] = stream

    try:
        if stream:
            with session.post(endpoint, json=payload, stream=True, timeout=3600) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    yield line.decode("utf-8")
        else:
            response = session.post(endpoint, json=payload, timeout=3600)
            response.raise_for_status()
            yield response.json()
    except requests.RequestException as err:
        yield {"error": f"Ollama request failed: {err}"}


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
