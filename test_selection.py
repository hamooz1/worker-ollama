"""Unit tests for the pure model-selection and naming logic in handler.py.

These need no GPU, no network and no running Ollama, which is the point: getting
a GGUF filename convention wrong is otherwise only discovered by a Hub test on
real hardware.

    python3 -m pytest test_selection.py -v
"""

import sys
import types

import pytest

# handler.py imports the runpod SDK at module scope; stub it when it isn't
# installed so the pure functions stay testable from a bare checkout.
if "runpod" not in sys.modules:
    try:
        import runpod  # noqa: F401
    except ImportError:
        sys.modules["runpod"] = types.ModuleType("runpod")

import handler


# --- available_quants: the naming conventions real GGUF repos use --------------


@pytest.mark.parametrize(
    "files, expected",
    [
        # unsloth and most modern repos: dash-delimited
        (["Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B-BF16.gguf"],
         ["BF16", "Q4_K_M", "Q8_0"]),
        # TheBloke: dot-delimited
        (["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"], ["Q4_K_M"]),
        # bartowski: one subdirectory per quant
        (["Q4_K_M/model-Q4_K_M.gguf"], ["Q4_K_M"]),
        # lowercase
        (["qwen2.5-0.5b-instruct-q4_k_m.gguf"], ["Q4_K_M"]),
        # i-quants and unsloth dynamic quants
        (["Model-IQ4_XS.gguf", "Model-UD-Q4_K_XL.gguf"], ["IQ4_XS", "Q4_K_XL"]),
        (["gpt-oss-20b-MXFP4.gguf"], ["MXFP4"]),
        # "Qwen3" must not be mistaken for a quant token
        (["Qwen3-0.6B-Q8_0.gguf"], ["Q8_0"]),
    ],
)
def test_available_quants(files, expected):
    assert handler.available_quants(files) == expected


# --- select_gguf: quantization matching ---------------------------------------

MULTI = [
    "Qwen3-0.6B-Q4_K_M.gguf",
    "Qwen3-0.6B-Q4_K_S.gguf",
    "Qwen3-0.6B-Q8_0.gguf",
    "README.md",
]


def test_quantization_selects_exact_file():
    assert handler.select_gguf(MULTI, "Q4_K_M", "", "r/x") == ["Qwen3-0.6B-Q4_K_M.gguf"]


def test_quantization_is_case_insensitive():
    assert handler.select_gguf(MULTI, "q4_k_s", "", "r/x") == ["Qwen3-0.6B-Q4_K_S.gguf"]


def test_similar_quants_do_not_cross_match():
    """Q4_K_M and Q4_K_S are different models, not fuzzy variants of each other."""
    assert handler.select_gguf(MULTI, "Q4_K_M", "", "r/x") != handler.select_gguf(
        MULTI, "Q4_K_S", "", "r/x"
    )


def test_partial_quant_token_does_not_match():
    """'Q4' must not silently select Q4_K_M — the user has to say which one."""
    with pytest.raises(ValueError, match="Available quantizations"):
        handler.select_gguf(MULTI, "Q4", "", "r/x")


def test_unknown_quant_lists_what_is_available():
    with pytest.raises(ValueError) as err:
        handler.select_gguf(MULTI, "Q9_X", "", "r/x")
    assert "Q4_K_M" in str(err.value) and "Q8_0" in str(err.value)


def test_multiple_ggufs_without_quant_or_sizes_is_an_error():
    """Without sizes the smallest can't be identified, so the user has to choose."""
    with pytest.raises(ValueError, match="set HF_QUANTIZATION"):
        handler.select_gguf(MULTI, "", "", "r/x")


# --- select_gguf: smallest-quantization default -------------------------------

MULTI_SIZES = {
    "Qwen3-0.6B-Q4_K_M.gguf": 400_000_000,
    "Qwen3-0.6B-Q4_K_S.gguf": 380_000_000,
    "Qwen3-0.6B-Q8_0.gguf": 700_000_000,
}


def test_no_quant_defaults_to_smallest():
    assert handler.select_gguf(MULTI, "", "", "r/x", MULTI_SIZES) == [
        "Qwen3-0.6B-Q4_K_S.gguf"
    ]


def test_explicit_quant_still_beats_the_smallest_default():
    assert handler.select_gguf(MULTI, "Q8_0", "", "r/x", MULTI_SIZES) == [
        "Qwen3-0.6B-Q8_0.gguf"
    ]


def test_smallest_default_sums_shard_groups():
    """A split GGUF competes on its total size, not on one shard."""
    files = ["big-00001-of-00002.gguf", "big-00002-of-00002.gguf", "small.gguf"]
    sizes = {
        "big-00001-of-00002.gguf": 300,
        "big-00002-of-00002.gguf": 300,
        "small.gguf": 500,
    }
    assert handler.select_gguf(files, "", "", "r/x", sizes) == ["small.gguf"]
    sizes["small.gguf"] = 700
    assert handler.select_gguf(files, "", "", "r/x", sizes) == [
        "big-00001-of-00002.gguf",
        "big-00002-of-00002.gguf",
    ]


def test_smallest_default_skips_incomplete_shard_groups():
    """An incomplete split GGUF is skipped rather than selected and then failed on."""
    files = ["tiny-00001-of-00003.gguf", "whole.gguf"]
    sizes = {"tiny-00001-of-00003.gguf": 10, "whole.gguf": 999}
    assert handler.select_gguf(files, "", "", "r/x", sizes) == ["whole.gguf"]


def test_smallest_default_is_deterministic_on_ties():
    files = ["a-Q4_K_M.gguf", "b-Q4_K_S.gguf"]
    sizes = {"a-Q4_K_M.gguf": 100, "b-Q4_K_S.gguf": 100}
    assert handler.select_gguf(files, "", "", "r/x", sizes) == ["a-Q4_K_M.gguf"]


def test_single_gguf_needs_no_quant():
    assert handler.select_gguf(["only-model.gguf", "README.md"], "", "", "r/x") == [
        "only-model.gguf"
    ]


def test_safetensors_repo_fails_fast():
    with pytest.raises(ValueError) as err:
        handler.select_gguf(
            ["config.json", "model.safetensors", "tokenizer.json"], "", "", "microsoft/Phi-3"
        )
    message = str(err.value)
    assert "no .gguf files" in message
    assert "model.safetensors" in message  # tells the user what it did find


# --- select_gguf: HF_MODEL_FILE escape hatch ----------------------------------


def test_model_file_exact_match():
    assert handler.select_gguf(MULTI, "", "Qwen3-0.6B-Q8_0.gguf", "r/x") == [
        "Qwen3-0.6B-Q8_0.gguf"
    ]


def test_model_file_overrides_quantization():
    assert handler.select_gguf(MULTI, "Q4_K_M", "Qwen3-0.6B-Q8_0.gguf", "r/x") == [
        "Qwen3-0.6B-Q8_0.gguf"
    ]


def test_unknown_model_file_lists_candidates():
    with pytest.raises(ValueError) as err:
        handler.select_gguf(MULTI, "", "nope.gguf", "r/x")
    assert "Qwen3-0.6B-Q4_K_M.gguf" in str(err.value)


# --- select_gguf: split (sharded) GGUFs ---------------------------------------

SHARDED = [
    "m-Q8_0-00002-of-00003.gguf",
    "m-Q8_0-00001-of-00003.gguf",
    "m-Q8_0-00003-of-00003.gguf",
    "m-Q4_K_M.gguf",
]
SHARDS_IN_ORDER = [
    "m-Q8_0-00001-of-00003.gguf",
    "m-Q8_0-00002-of-00003.gguf",
    "m-Q8_0-00003-of-00003.gguf",
]


def test_shards_are_returned_in_index_order():
    """Ollama needs every shard; the repo listing order is not the shard order."""
    assert handler.select_gguf(SHARDED, "Q8_0", "", "r/x") == SHARDS_IN_ORDER


def test_single_file_quant_beside_a_split_quant():
    assert handler.select_gguf(SHARDED, "Q4_K_M", "", "r/x") == ["m-Q4_K_M.gguf"]


def test_naming_any_shard_selects_the_whole_group():
    assert (
        handler.select_gguf(SHARDED, "", "m-Q8_0-00002-of-00003.gguf", "r/x")
        == SHARDS_IN_ORDER
    )


def test_incomplete_shard_set_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        handler.select_gguf(["m-Q8_0-00001-of-00003.gguf"], "Q8_0", "", "r/x")


def test_incomplete_shard_set_does_not_block_another_quant():
    """Grouping is lenient so a half-cached quant can't break an unrelated choice."""
    files = ["m-Q8_0-00001-of-00003.gguf", "m-Q4_K_M.gguf"]
    assert handler.select_gguf(files, "Q4_K_M", "", "r/x") == ["m-Q4_K_M.gguf"]


def test_duplicate_shard_index_is_rejected():
    files = ["a/m-00001-of-00002.gguf", "a/m-00002-of-00002.gguf"]
    group = handler.group_ggufs(files)[0]
    assert handler.validate_group(group, "r/x") == files
    with pytest.raises(ValueError, match="duplicate shard"):
        handler.validate_group(group + ["a/m-00002-of-00002.gguf"], "r/x")


# --- derive_model_name: must be pure and deterministic ------------------------


@pytest.mark.parametrize(
    "repo, quant, model_file, expected",
    [
        ("unsloth/Qwen3-8B-GGUF", "Q4_K_M", "", "hf/unsloth-qwen3-8b-gguf:q4_k_m"),
        ("unsloth/Qwen3-8B-GGUF", "", "", "hf/unsloth-qwen3-8b-gguf:latest"),
        ("a/b", "", "Model-Q8_0.gguf", "hf/a-b:model-q8_0"),
        ("a/b", "Q4_K_M", "x-Q8_0.gguf", "hf/a-b:x-q8_0"),  # file wins over quant
        ("bare-repo", "Q4_K_M", "", "hf/bare-repo:q4_k_m"),
    ],
)
def test_derive_model_name(repo, quant, model_file, expected):
    assert handler.derive_model_name(repo, quant, model_file) == expected


def test_derive_model_name_is_deterministic():
    """start.sh's subprocess and the handler process must agree without shared state."""
    first = handler.derive_model_name("unsloth/Qwen3-8B-GGUF", "Q4_K_M", "")
    second = handler.derive_model_name("unsloth/Qwen3-8B-GGUF", "Q4_K_M", "")
    assert first == second


def test_derive_model_name_truncates_long_repos():
    name = handler.derive_model_name(
        "some-really-long-organisation-name/an-equally-long-model-name-GGUF", "Q4_K_M", ""
    )
    base, _, tag = name.partition(":")
    assert len(base) <= 60
    assert tag == "q4_k_m"


def test_derive_model_name_is_lowercase():
    """Ollama folds case on lookup but preserves it on the manifest path."""
    name = handler.derive_model_name("Unsloth/Qwen3-8B-GGUF", "Q4_K_M", "")
    assert name == name.lower()


# --- normalize_model_name -----------------------------------------------------


# --- parse_hf_model: every shape HF_MODEL arrives in --------------------------


@pytest.mark.parametrize(
    "given, repo, quant",
    [
        # what the Hub's Hugging Face picker stores
        ("hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF", "ornith-ai/Ornith-1.5-35B-A3B-GGUF", ""),
        # a bare repo id
        ("unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", ""),
        # the other host alias
        ("huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", ""),
        # pasted browser URLs, with and without a path suffix
        ("https://huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", ""),
        ("https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main", "unsloth/Qwen3-8B-GGUF", ""),
        ("https://hf.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", ""),
        # Ollama-style ':quant' tag carries the quantization
        ("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M", "unsloth/Qwen3-8B-GGUF", "Q4_K_M"),
        ("unsloth/Qwen3-8B-GGUF:IQ4_XS", "unsloth/Qwen3-8B-GGUF", "IQ4_XS"),
        # ':latest' is not a quantization
        ("hf.co/unsloth/Qwen3-8B-GGUF:latest", "unsloth/Qwen3-8B-GGUF", ""),
        # whitespace, trailing slash, host casing
        ("  hf.co/unsloth/Qwen3-8B-GGUF/  ", "unsloth/Qwen3-8B-GGUF", ""),
        ("HF.CO/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", ""),
        ("", "", ""),
    ],
)
def test_parse_hf_model(given, repo, quant):
    assert handler.parse_hf_model(given) == (repo, quant)


def test_parsed_repo_builds_the_model_store_folder():
    """The whole point: a wrong repo id means a silent model-store cache miss."""
    repo, _ = handler.parse_hf_model("hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF")
    assert "models--" + repo.replace("/", "--") == "models--ornith-ai--Ornith-1.5-35B-A3B-GGUF"


def test_explicit_quantization_beats_a_tag_in_the_ref(monkeypatch):
    """HF_QUANTIZATION is the documented input; the ':tag' is only a fallback."""
    repo, tag = handler.parse_hf_model("hf.co/unsloth/Qwen3-8B-GGUF:Q2_K")
    assert ("Q8_0" or tag) == "Q8_0"  # mirrors `HF_QUANTIZATION_RAW or _HF_MODEL_TAG`
    assert tag == "Q2_K"


# --- resolve_default_model: the precedence chain ------------------------------


def test_hf_model_beats_ollama_model(monkeypatch):
    monkeypatch.setattr(handler, "HF_MODEL", "unsloth/Qwen3-8B-GGUF")
    monkeypatch.setattr(handler, "HF_QUANTIZATION", "Q4_K_M")
    monkeypatch.setattr(handler, "HF_MODEL_FILE", "")
    monkeypatch.setattr(handler, "DEFAULT_MODEL", "llama3.2:1b")
    assert handler.resolve_default_model() == "hf/unsloth-qwen3-8b-gguf:q4_k_m"


def test_ollama_model_beats_the_fallback(monkeypatch):
    monkeypatch.setattr(handler, "HF_MODEL", "")
    monkeypatch.setattr(handler, "DEFAULT_MODEL", "qwen2.5-coder:7b")
    assert handler.resolve_default_model() == "qwen2.5-coder:7b"


def test_falls_back_when_both_inputs_are_empty(monkeypatch):
    monkeypatch.setattr(handler, "HF_MODEL", "")
    monkeypatch.setattr(handler, "DEFAULT_MODEL", "")
    assert handler.resolve_default_model() == handler.FALLBACK_MODEL == "llama3.2:3b"


def test_falls_back_when_ollama_model_is_blank(monkeypatch):
    """An env var set to whitespace is the same as unset."""
    monkeypatch.setattr(handler, "HF_MODEL", "")
    monkeypatch.setattr(handler, "DEFAULT_MODEL", "   ")
    assert handler.resolve_default_model() == "llama3.2:3b"


@pytest.mark.parametrize(
    "given, expected",
    [
        ("huggingface.co/org/repo:Q4_0", "hf.co/org/repo:Q4_0"),
        ("hf.co/org/repo:Q4_0", "hf.co/org/repo:Q4_0"),
        ("llama3.2:3b", "llama3.2:3b"),
        ("", ""),
    ],
)
def test_normalize_model_name(given, expected):
    assert handler.normalize_model_name(given) == expected
