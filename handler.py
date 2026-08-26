import hashlib
import json
import os
import re

import requests
import runpod

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS", "/root/.ollama/models")

# Hugging Face inputs. These are NOT the same thing as OLLAMA_MODEL: HF_MODEL is a
# Hugging Face repo id, OLLAMA_MODEL is an Ollama model reference. HF_MODEL wins.
HF_MODEL = os.environ.get("HF_MODEL", "").strip().strip("/")
HF_QUANTIZATION = os.environ.get("HF_QUANTIZATION", "").strip()
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "").strip()
# Where Runpod's model store mounts its prefilled Hugging Face cache. There is no
# env var injected by the platform to discover this, so it is a documented path
# with an override in case it moves.
RUNPOD_MODEL_CACHE_DIR = os.environ.get(
    "RUNPOD_MODEL_CACHE_DIR", "/runpod-volume/huggingface-cache/hub"
)
OLLAMA_TEMPLATE = os.environ.get("OLLAMA_TEMPLATE", "")
# Used only when neither HF_MODEL nor OLLAMA_MODEL is set, so the worker still
# serves something out of the box rather than erroring on the first request.
FALLBACK_MODEL = "llama3.2:3b"

# Quantization tokens as they appear in GGUF filenames. The boundary sets matter:
# quant tokens contain underscores, so splitting on "_" would be wrong, and a
# trailing "_" must not match (otherwise "Q4" would select "Q4_K_M").
QUANT_RE = re.compile(
    r"(?:^|[-_./])(I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4)(?=[-./]|$)", re.I
)
# Mirrors Ollama's own splitGGUFNameRe. Matching a different pattern than the
# server does is how you end up with a silently broken multi-layer manifest.
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)

session = requests.Session()


def get_local_models():
    response = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def normalize_model_name(model):
    """Ollama only knows the 'hf.co' registry host.

    'huggingface.co/...' parses as a different host, so its manifest would be
    written somewhere Ollama never looks it up.
    """
    model = (model or "").strip()
    if model.startswith("huggingface.co/"):
        return "hf.co/" + model[len("huggingface.co/") :]
    return model


def _sanitize(part):
    """Ollama name parts allow [A-Za-z0-9_.-], but lookups fold case while the
    on-disk manifest path preserves it, so lowercase is the only safe form."""
    part = re.sub(r"[^a-z0-9._-]+", "-", part.lower()).strip("-._")
    return part or "model"


def derive_model_name(repo_id, quantization, model_file):
    """The Ollama name a Hugging Face repo is registered under.

    Pure function of the env inputs on purpose: start.sh prepares the model in a
    short-lived subprocess and then execs the handler as a separate process, so
    both must derive the same name without sharing state. In particular the tag
    comes from the *inputs*, never from whichever file selection picked, because
    selection needs to list the repo.
    """
    org, _, name = repo_id.strip("/").partition("/")
    base = _sanitize(f"{org}-{name}" if name else org)
    if len(base) > 55:
        digest = hashlib.sha256(repo_id.encode()).hexdigest()[:6]
        base = base[:48].rstrip("-._") + "-" + digest
    if model_file:
        tag = _sanitize(re.sub(r"\.gguf$", "", os.path.basename(model_file), flags=re.I))
    elif quantization:
        tag = _sanitize(quantization)
    else:
        tag = "latest"
    return f"hf/{base}:{tag[:60].rstrip('-._') or 'latest'}"


def resolve_default_model():
    """The single place HF_MODEL's precedence over OLLAMA_MODEL is decided.

    Precedence: HF_MODEL, then OLLAMA_MODEL, then FALLBACK_MODEL.
    """
    if HF_MODEL:
        return derive_model_name(HF_MODEL, HF_QUANTIZATION, HF_MODEL_FILE)
    return normalize_model_name(DEFAULT_MODEL) or FALLBACK_MODEL


def _hf_cache_roots():
    """Runpod's prefilled model store first.

    That mount may be read-only, so the read path is deliberately separate from
    the directory fallback downloads are written to.
    """
    roots = []
    for root in (
        RUNPOD_MODEL_CACHE_DIR,
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
    ):
        if root and root not in roots and os.path.isdir(root):
            roots.append(root)
    return roots


def find_cached_snapshot(repo_id):
    """Locate a Hugging Face hub snapshot dir, as Runpod's model store lays it out."""
    folder = "models--" + repo_id.strip("/").replace("/", "--")
    for root in _hf_cache_roots():
        snapshots = os.path.join(root, folder, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        ref = os.path.join(root, folder, "refs", "main")
        if os.path.isfile(ref):
            with open(ref) as f:
                candidate = os.path.join(snapshots, f.read().strip())
            if os.path.isdir(candidate):
                return candidate
        versions = sorted(
            d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))
        )
        if versions:
            return os.path.join(snapshots, versions[0])
    return None


def list_snapshot_files(snapshot_dir):
    """Repo-relative paths of files that are really there.

    Snapshot entries are symlinks into the cache's blobs/ dir, and a partially
    filled cache leaves dangling ones behind.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(snapshot_dir, followlinks=True):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.isfile(os.path.realpath(full)):
                found.append(os.path.relpath(full, snapshot_dir))
    return sorted(found)


def _human_size(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.0f} {unit}" if unit in ("B", "KiB") else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def _stem(path):
    return re.sub(r"\.gguf$", "", path, flags=re.I)


def _shard_key(path):
    match = SHARD_RE.match(os.path.basename(path))
    if not match:
        return None
    return (os.path.dirname(path), match.group(1), match.group(3))


def available_quants(gguf_files):
    """Quantizations present in a repo, for error messages that tell users what to pick."""
    return sorted(
        {m.group(1).upper() for f in gguf_files for m in QUANT_RE.finditer(_stem(f))}
    )


def group_ggufs(gguf_files):
    """Collapse split-GGUF shards into one entry each, preserving listing order.

    Grouping is deliberately lenient here — an incomplete shard set for one
    quantization must not break selecting a different one.
    """
    groups = []
    index_of = {}
    for path in gguf_files:
        key = _shard_key(path)
        if key is None:
            groups.append([path])
            continue
        if key not in index_of:
            index_of[key] = len(groups)
            groups.append([])
        groups[index_of[key]].append(path)
    return groups


def validate_group(group, repo_id):
    """Order and completeness-check a selected group.

    Ollama needs every shard of a split GGUF, keyed by the exact filename
    llama.cpp wrote; handing it only the first shard is an error.
    """
    if _shard_key(group[0]) is None:
        return group
    total = int(SHARD_RE.match(os.path.basename(group[0])).group(3))
    found = {}
    for path in group:
        index = int(SHARD_RE.match(os.path.basename(path)).group(2))
        if index in found:
            raise ValueError(
                f"'{repo_id}' has duplicate shard {index} for "
                f"'{os.path.basename(group[0])}'."
            )
        found[index] = path
    missing = [i for i in range(1, total + 1) if i not in found]
    if missing:
        raise ValueError(
            f"Split GGUF '{os.path.basename(group[0])}' in '{repo_id}' is incomplete — "
            f"missing shard(s) {missing} of {total}. Ollama needs every shard."
        )
    return [found[i] for i in range(1, total + 1)]


def select_gguf(files, quantization, model_file, repo_id, sizes=None):
    """Repo-relative path(s) of the GGUF to load: one file, or every shard of a split GGUF.

    `sizes` maps repo-relative paths to byte sizes and is what makes the
    no-quantization default possible: without an explicit HF_QUANTIZATION the
    smallest variant wins.
    """
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise ValueError(
            f"Hugging Face repo '{repo_id}' contains no .gguf files, so Ollama cannot "
            f"run it. HF_MODEL must point at a GGUF repo — those are usually named "
            f"'<model>-GGUF', e.g. 'unsloth/Qwen3-8B-GGUF'. To run a safetensors model, "
            f"use a GGUF conversion of it, or set OLLAMA_MODEL to an Ollama library "
            f"model instead. Files in the repo: {sorted(files)[:15]}"
        )

    if model_file:
        wanted = os.path.basename(model_file.strip().lstrip("./")).lower()
        for candidate in ggufs:
            if candidate == model_file.strip() or os.path.basename(candidate).lower() == wanted:
                key = _shard_key(candidate)
                group = next(
                    g for g in group_ggufs(ggufs) if _shard_key(g[0]) == key and (
                        key is not None or g[0] == candidate
                    )
                )
                return validate_group(group, repo_id)
        raise ValueError(
            f"HF_MODEL_FILE '{model_file}' is not in '{repo_id}'. "
            f"Available GGUF files: {ggufs[:20]}"
        )

    groups = group_ggufs(ggufs)

    if quantization:
        pattern = re.compile(
            rf"(?:^|[-_./]){re.escape(quantization)}(?=[-./]|$)", re.I
        )
        matches = [g for g in groups if pattern.search(_stem(g[0]))]
        if len(matches) == 1:
            return validate_group(matches[0], repo_id)
        if not matches:
            raise ValueError(
                f"No GGUF in '{repo_id}' matches HF_QUANTIZATION='{quantization}'. "
                f"Available quantizations: {available_quants(ggufs)}. "
                f"GGUF files: {[g[0] for g in groups][:20]}"
            )
        raise ValueError(
            f"HF_QUANTIZATION='{quantization}' is ambiguous in '{repo_id}' — it matches "
            f"{[g[0] for g in matches][:10]}. Set HF_MODEL_FILE to the exact filename."
        )

    if len(groups) == 1:
        return validate_group(groups[0], repo_id)

    # No quantization asked for: take the smallest variant, so the default fits
    # the smallest GPU and the shortest cold start.
    if sizes:
        def group_size(group):
            return sum(sizes.get(path, 0) for path in group)

        for group in sorted(groups, key=lambda g: (group_size(g), g[0])):
            if group_size(group) <= 0:
                continue
            try:
                chosen = validate_group(group, repo_id)
            except ValueError:
                continue  # incomplete split GGUF — try the next size up
            print(
                f"HF_QUANTIZATION not set — defaulting to the smallest GGUF in "
                f"'{repo_id}': {os.path.basename(chosen[0])} "
                f"({_human_size(group_size(group))}). "
                f"Available quantizations: {available_quants(ggufs)}",
                flush=True,
            )
            return chosen

    raise ValueError(
        f"'{repo_id}' contains {len(groups)} GGUF variants and their sizes could not be "
        f"determined, so the smallest can't be picked automatically — set "
        f"HF_QUANTIZATION to choose one. Available quantizations: {available_quants(ggufs)}."
    )


def acquire_gguf(repo_id, quantization, model_file):
    """Absolute path(s) of the GGUF to register, preferring Runpod's model store.

    The model store is the intended path: Runpod prefills it before the worker
    starts and doesn't bill for the download. Fetching from Hugging Face directly
    is the fallback, and says so loudly because it costs cold-start time.
    """
    snapshot = find_cached_snapshot(repo_id)
    if snapshot:
        available = list_snapshot_files(snapshot)
        sizes = {}
        for rel in available:
            if rel.lower().endswith(".gguf"):
                try:
                    sizes[rel] = os.path.getsize(os.path.realpath(os.path.join(snapshot, rel)))
                except OSError:
                    pass
        selected = select_gguf(available, quantization, model_file, repo_id, sizes)
        paths = [os.path.realpath(os.path.join(snapshot, rel)) for rel in selected]
        if all(os.path.isfile(p) for p in paths):
            print(f"[ModelStore] Using snapshot {snapshot}", flush=True)
            for path in paths:
                size = _human_size(os.path.getsize(path))
                print(f"[ModelStore]   {os.path.basename(path)} ({size})", flush=True)
            return paths
        print(
            f"[ModelStore] Snapshot {snapshot} is missing files for the selected "
            f"quantization — falling back to download",
            flush=True,
        )
    else:
        print(
            f"WARN: no cached snapshot for '{repo_id}' under {RUNPOD_MODEL_CACHE_DIR}.\n"
            f"      Downloading from Hugging Face instead (billed cold-start time).\n"
            f"      To use Runpod's model store, set the endpoint's Model field to "
            f"'{repo_id}'.",
            flush=True,
        )

    # Imported lazily so endpoints that only use OLLAMA_MODEL never pay for it.
    # list_repo_tree rather than list_repo_files: it returns sizes in the same
    # call, which the smallest-quantization default needs.
    from huggingface_hub import hf_hub_download, list_repo_tree

    token = HF_TOKEN or None
    entries = list(list_repo_tree(repo_id, recursive=True, token=token))
    available = [e.path for e in entries]
    sizes = {
        e.path: e.size
        for e in entries
        if e.path.lower().endswith(".gguf") and getattr(e, "size", None)
    }
    selected = select_gguf(available, quantization, model_file, repo_id, sizes)
    paths = []
    for rel in selected:
        print(f"Downloading {repo_id}/{rel}", flush=True)
        paths.append(os.path.realpath(hf_hub_download(repo_id, rel, token=token)))
    return paths


def file_digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def blob_present(digest):
    response = session.head(f"{OLLAMA_BASE_URL}/api/blobs/{digest}", timeout=30)
    return response.status_code == 200


def link_blob(path, digest):
    """Hard-link a GGUF into Ollama's blob store instead of uploading it.

    Ollama resolves blobs with a bare stat and keeps no index, so a correctly
    named hard link is indistinguishable from an upload — and moves no bytes.
    Returns False when the file and the store are on different filesystems.
    """
    blobs_dir = os.path.join(OLLAMA_MODELS_DIR, "blobs")
    final = os.path.join(blobs_dir, digest.replace(":", "-"))
    temp = final + ".link"
    try:
        os.makedirs(blobs_dir, exist_ok=True)
        if os.path.lexists(temp):
            os.remove(temp)
        os.link(path, temp)
        os.replace(temp, final)
        return True
    except OSError as err:
        print(f"Hard link into {blobs_dir} failed ({err}) — uploading blob instead", flush=True)
        try:
            os.remove(temp)
        except OSError:
            pass
        return False


def upload_blob(path, digest):
    with open(path, "rb") as f:
        response = session.post(f"{OLLAMA_BASE_URL}/api/blobs/{digest}", data=f, timeout=7200)
    response.raise_for_status()


def warn_if_no_template(model):
    """A GGUF without an embedded chat template yields plausible-looking but
    malformed chat output, which a smoke test happily passes."""
    try:
        response = session.post(f"{OLLAMA_BASE_URL}/api/show", json={"model": model}, timeout=60)
        response.raise_for_status()
        if not (response.json().get("template") or "").strip():
            print(
                f"WARN: '{model}' has no chat template embedded in the GGUF. Chat "
                f"responses may be malformed — set OLLAMA_TEMPLATE, or pass 'template' "
                f"in the request input.",
                flush=True,
            )
    except (requests.RequestException, ValueError):
        pass


def create_model_from_gguf(model, paths):
    """Register local GGUF file(s) with Ollama under `model`."""
    files = {}
    for path in paths:
        digest = file_digest(path)
        if not blob_present(digest) and not link_blob(path, digest):
            upload_blob(path, digest)
        files[os.path.basename(path)] = digest

    payload = {"model": model, "files": files, "stream": False}
    if OLLAMA_TEMPLATE:
        payload["template"] = OLLAMA_TEMPLATE
    print(f"Registering '{model}' from {sorted(files)}", flush=True)
    response = session.post(f"{OLLAMA_BASE_URL}/api/create", json=payload, timeout=7200)
    response.raise_for_status()

    # /api/create answers 200 and reports failures inside the body, so the status
    # code alone proves nothing.
    error = None
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("error"):
            error = event["error"]
    if error:
        raise ValueError(f"ollama create failed for '{model}': {error}")

    warn_if_no_template(model)


def pull_hf_model_with_token(model):
    """Pull an hf.co model using an HF access token (gated/private repos).

    Ollama's native pull authenticates with an SSH key, which serverless
    workers can't practically register. Instead, fetch the same manifest and
    blobs from Hugging Face's Ollama-compatible registry with bearer auth and
    write them straight into Ollama's model store.
    """
    if not model.startswith("hf.co/"):
        raise ValueError(f"expected an 'hf.co/' model reference, got '{model}'")
    ref = model.split("/", 1)[1]  # strip "hf.co/"
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
    model = normalize_model_name(model)
    local = get_local_models()
    if model in local or f"{model}:latest" in local:
        return
    if HF_TOKEN and model.startswith("hf.co/"):
        pull_hf_model_with_token(model)
        return
    response = session.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model, "stream": False},
        timeout=3600,
    )
    response.raise_for_status()


def ensure_default_model():
    """Prepare the model the endpoint is configured for.

    Idempotent, so start.sh can call it at boot and the handler can call it again
    on the first request if that startup attempt failed.
    """
    model = resolve_default_model()
    if not model:
        return ""
    local = get_local_models()
    if model in local or f"{model}:latest" in local:
        print(f"Model already present: {model}", flush=True)
        return model
    if HF_MODEL:
        create_model_from_gguf(model, acquire_gguf(HF_MODEL, HF_QUANTIZATION, HF_MODEL_FILE))
    else:
        ensure_model(model)
    return model


def handler(job):
    job_input = job.get("input") or {}

    requested = job_input.get("model")
    model = normalize_model_name(requested) if requested else resolve_default_model()
    if not model:
        yield {
            "error": (
                "The 'model' value in the request input is blank. Omit it to use the "
                "model the endpoint is configured with, or pass a valid model name."
            )
        }
        return

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if not messages and not prompt:
        yield {"error": "Provide either 'messages' (chat) or 'prompt' (completion) in input."}
        return

    try:
        if requested:
            ensure_model(model)
        else:
            ensure_default_model()
    except (requests.RequestException, ValueError, OSError) as err:
        yield {"error": f"Failed to prepare model '{model}': {err}"}
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
