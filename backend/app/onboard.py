# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Auto-onboard a model from a Hugging Face id (Tier 1).

The bottleneck for supporting a newly-launched model is the hand-written
catalog.py entry (hf id, template, LoRA targets, cutoff, instance, gated flag) —
and a wrong template silently fails the job. This module removes that by
deriving the FACTS authoritatively from the model's config + HF metadata, and
matching the chat template by architecture. Facts are read, never guessed; the
one fuzzy step (template) is CONFIRMED by a 10-step smoke test before the model
is trusted (see orchestrate.launch_smoke_test / race wiring).

Custom (onboarded) models persist in the state store so they survive restarts
and are visible to every Lambda; catalog.get_model/list_models merge them with
the built-in CATALOG.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from .catalog import ModelSpec, _instance_for  # reuse the size→instance bands
from .secrets import get_hf_token
from .store import get_store

# Custom models live as a single root JSON doc {modelId: spec-dict, ...}.
_CUSTOM_FILE = "custom_models.json"

HF_API = "https://huggingface.co/api/models/{repo}"
HF_CONFIG = "https://huggingface.co/{repo}/resolve/main/config.json"

# A Hugging Face repo id is exactly "<org>/<name>" where each segment is limited
# to word chars, dot and dash. Validating against this allowlist BEFORE the value
# is interpolated into a URL closes the tool-misuse / SSRF surface (OWASP-Agentic
# ASI-02 / CWE-918): it rejects path traversal ('../'), embedded credentials or
# host overrides ('@evil.com'), CRLF, query/fragment injection, and protocol
# smuggling — none of which can appear in a legitimate repo id.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def _validate_repo(repo: str) -> str:
    """Return a clean HF repo id or raise ValueError. The single trust boundary
    for every value that flows into an HF URL."""
    if not isinstance(repo, str):
        raise ValueError("repo must be a string")
    repo = repo.strip()
    if not repo or len(repo) > 200 or not _HF_REPO_RE.match(repo):
        raise ValueError(
            f"invalid Hugging Face repo id '{repo}': expected '<org>/<name>' "
            "using letters, digits, '.', '_' or '-'"
        )
    return repo

# Architecture → LLaMA-Factory template. This is the fuzzy mapping (confirmed by
# the smoke test). Keys are matched as substrings of the `architectures` entry
# in config.json (case-insensitive), longest-match-wins. Verified against the
# live LLaMA-Factory README supported-models table.
_ARCH_TEMPLATE: dict[str, str] = {
    "qwen3moe": "qwen3",
    "qwen3": "qwen3",
    "qwen2": "qwen",
    "llama": "llama3",  # Llama-3.x; Llama-2 would need "llama2" (rare now)
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma2": "gemma2",
    "gemma": "gemma",
    "phi3": "phi",
    "phimoe": "phi",
    "phi": "phi4",  # Phi-4 family; phi-3.5-mini uses "phi" (caught by phi3 above)
    "glm4": "glm4",
    "chatglm": "glm4",
    "internlm2": "intern2",
    "minicpm": "cpm4",
    "granite": "granite3",
    "deepseek": "deepseekr1",
    "falcon": "falcon_h1",
}


# Templates LLaMA-Factory 0.9.4 actually ships (from its template registry). A
# matched/edited template MUST be one of these — otherwise the engine rejects it
# at config parse (the #1 onboarding failure), so we validate deterministically
# and for free before ever launching a job.
KNOWN_TEMPLATES: set[str] = {
    "qwen", "qwen3", "llama2", "llama3", "mistral", "gemma", "gemma2", "phi", "phi4",
    "phi4_mini", "glm4", "glmz1", "chatglm3", "intern2", "cpm4", "granite3",
    "deepseek", "deepseekr1", "deepseek3", "falcon", "falcon_h1", "yi", "baichuan2",
    "phi_small", "gpt_oss", "gemma3", "llama2", "llama4", "cpm", "cpm3", "intern",
    "default", "alpaca", "vicuna", "chatml", "zephyr",
    # qwen3_nothink: the NON-thinking Qwen3 variant template (e.g. Qwen3-*-Instruct-2507)
    # — plain ChatML, no <think> injection. Verified present in BOTH 0.9.4 + 0.9.5
    # image manifests. Using "qwen3" for a non-thinking model wrongly teaches empty
    # <think></think> tokens during SFT.
    "qwen3_nothink",
    # Templates that only the `latest` image (LLaMA-Factory 0.9.5) ships — used by
    # models pinned to image_tag="latest". NOT in the `stable` 0.9.4 image, which is
    # exactly why those models declare `latest`. (Verified against the image
    # capability manifests.)
    "lfm2", "lfm2_vl",
    # qwen3_5 / qwen3_5_nothink: the Qwen3.5 family template (distinct tool_format +
    # qwen3_vl mm_plugin). 0.9.5-only (absent from 0.9.4) — models using it must pin
    # image_tag="latest".
    "qwen3_5", "qwen3_5_nothink",
}


def is_known_template(template: str | None) -> bool:
    return bool(template) and template in KNOWN_TEMPLATES


def _match_template(architectures: list[str], model_type: str,
                    repo: str = "") -> tuple[str | None, str]:
    """Best-effort template match. Returns (template, how). template is None if
    nothing matched confidently (the UI then asks the user to pick / the smoke
    test will fail fast).

    The chat template reflects how a model was POST-TRAINED, not its base
    architecture — so for families whose prompt format differs from what their base
    arch (or coarse family) implies, match the REPO NAME first:
      • DeepSeek-R1-Distill-{Llama,Qwen} report LlamaForCausalLM/Qwen2ForCausalLM
        (→ would wrongly match llama3/qwen) but use DeepSeek's R1 reasoning format
        → `deepseekr1`.
      • Qwen3-*-Instruct-2507 are NON-thinking (plain ChatML, no <think>); the
        thinking `qwen3` template injects empty <think></think> tokens with loss
        during SFT → use `qwen3_nothink`.
      • Qwen3.5 is a distinct family (model_type qwen3_5, own tool_format + mm_plugin)
        → `qwen3_5` (the model must pin image_tag="latest"; 0.9.5-only).
    The built-in catalog hand-sets these; mirror them here so onboarding agrees."""
    name = repo.lower()
    if "r1-distill" in name or "r1_distill" in name or "deepseek-r1" in name:
        return "deepseekr1", "matched repo name 'DeepSeek-R1-Distill' (R1 reasoning format)"
    # Qwen3.5 family (check before the generic qwen3 arch match). Covers 'qwen3.5'
    # and 'qwen3_5' spellings; the 'nothink' suffix follows the non-thinking marker.
    if "qwen3.5" in name or "qwen3_5" in name:
        nothink = "instruct" in name and "thinking" not in name
        tmpl = "qwen3_5_nothink" if nothink else "qwen3_5"
        return tmpl, f"matched repo name 'Qwen3.5' (Qwen3.5 family → {tmpl}; needs latest image)"
    # Qwen3 NON-thinking instruct variants (e.g. Qwen3-4B-Instruct-2507): plain
    # ChatML, no <think> — must NOT use the thinking 'qwen3' template.
    if "qwen3" in name and "instruct" in name and ("2507" in name or "nothink" in name):
        return "qwen3_nothink", "matched repo name 'Qwen3 *-Instruct-2507' (non-thinking → qwen3_nothink)"
    hay = " ".join(architectures + [model_type]).lower()
    # longest key first so 'qwen3' beats 'qwen2' substring overlaps etc.
    for key in sorted(_ARCH_TEMPLATE, key=len, reverse=True):
        if key in hay:
            return _ARCH_TEMPLATE[key], f"matched architecture '{key}'"
    return None, "no confident template match — set manually or rely on smoke test"


def _http_json(url: str, token: str | None) -> dict[str, Any]:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "slm-platform-onboard")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (fixed host)
        return json.loads(resp.read().decode("utf-8"))


def _params_billions(meta: dict[str, Any], config: dict[str, Any]) -> float:
    """Best-effort parameter count in billions. HF `safetensors.total` is exact
    when present; else estimate from config dims; else 0 (caller can ask user)."""
    st = meta.get("safetensors") or {}
    total = st.get("total")
    if isinstance(total, (int, float)) and total > 0:
        return round(total / 1e9, 2)
    # Rough estimate from transformer dims if safetensors metadata is absent.
    h = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    vocab = config.get("vocab_size")
    if h and layers and vocab:
        approx = 12 * layers * h * h + 2 * vocab * h
        return round(approx / 1e9, 2)
    return 0.0


def _slug(repo: str) -> str:
    """Stable catalog id from a HF repo (org/name → name-lowercased-safe)."""
    name = repo.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.-]", "-", name).strip("-") or "model"


def probe_model(repo: str) -> dict[str, Any]:
    """Fetch HF metadata + config for `repo` and derive a draft ModelSpec's
    fields. Does NOT persist — the caller reviews + smoke-tests first. Raises
    ValueError with a clear message on a missing/private repo."""
    repo = _validate_repo(repo)  # reject anything that isn't a clean HF repo id
    token = get_hf_token()
    try:
        meta = _http_json(HF_API.format(repo=repo), token)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not fetch HF metadata for '{repo}': {e}")
    try:
        config = _http_json(HF_CONFIG.format(repo=repo), token)
    except Exception:
        config = {}  # some repos hide config; we degrade gracefully

    architectures = config.get("architectures") or []
    model_type = config.get("model_type", "")
    template, how = _match_template(architectures, model_type, repo=repo)
    params_b = _params_billions(meta, config)
    cutoff = config.get("max_position_embeddings") or 2048
    # Clamp an absurd advertised context to a trainable default (e.g. Qwen3's
    # 262k) — the user can raise it; keeps the smoke test cheap.
    cutoff = min(int(cutoff), 8192)
    gated = bool(meta.get("gated"))  # 'auto'/'manual'/False → truthy if gated

    return {
        "id": _slug(repo),
        "hfModelId": repo,
        "displayName": repo.split("/")[-1],
        "family": (repo.split("/")[-1].split("-")[0] or "Custom").title(),
        "template": template,
        "templateMatch": how,
        "templateKnown": is_known_template(template),
        "knownTemplates": sorted(KNOWN_TEMPLATES),
        "paramsB": params_b,
        "defaultCutoffLen": cutoff,
        "suggestedInstance": _instance_for(params_b or 8.0),
        "gated": gated,
        "architectures": architectures,
        "modelType": model_type,
    }


def _custom() -> dict[str, Any]:
    return get_store().read_root_json(_CUSTOM_FILE)


def list_custom_models() -> list[dict[str, Any]]:
    return list(_custom().values())


def get_custom_model(model_id: str) -> ModelSpec | None:
    d = _custom().get(model_id)
    if not d:
        return None
    return _spec_from_dict(d)


def _spec_from_dict(d: dict[str, Any]) -> ModelSpec:
    band = "tiny" if d["paramsB"] <= 2 else "small" if d["paramsB"] <= 4 else "mid" if d["paramsB"] <= 10 else "large"
    return ModelSpec(
        id=d["id"],
        display_name=d["displayName"],
        hf_model_id=d["hfModelId"],
        template=d["template"],
        family=d.get("family", "Custom"),
        tier=band,
        params_b=d["paramsB"],
        default_cutoff_len=d["defaultCutoffLen"],
        suggested_instance=d["suggestedInstance"],
        gated=d.get("gated", False),
        notes=d.get("notes", "Auto-onboarded from Hugging Face."),
        image_tag=d.get("imageTag", "stable"),
        # A custom model can carry a serverless tag (its SageMaker Public Hub id),
        # so a NEW serverless-customizable model discovered on the hub can be
        # onboarded WITH the serverless engine already enabled — not just the
        # LLaMA-Factory path. Empty for the common HF-only onboard.
        serverless_model_id=d.get("serverlessModelId", ""),
        # Onboarded repos do NOT get repo-side code execution by default. An
        # arbitrary HF repo id is untrusted input: its config.json can `auto_map`
        # to modeling code that then runs unsandboxed in the training job, the
        # eval container and the exported endpoint under this deployment's role.
        # Opting in is a deliberate per-model act, so the default here is False
        # (matching ModelSpec) rather than the curated catalog's True.
        trust_remote_code=bool(d.get("trustRemoteCode", False)),
    )


def save_custom_model(spec_dict: dict[str, Any]) -> None:
    """Persist an onboarded model (after smoke test). Keyed by catalog id."""
    if not spec_dict.get("template"):
        raise ValueError("cannot save a model without a chat template")
    # A serverless tag (Public Hub id), if supplied, is validated at the trust
    # boundary before persistence — same allowlist the overlay path uses — so a
    # malformed id can't reach a launch spec via the custom record.
    serverless_id = (spec_dict.get("serverlessModelId") or "").strip()
    if serverless_id:
        from .serverless_catalog import _validate_hub_id

        serverless_id = _validate_hub_id(serverless_id)
    cur = _custom()
    cur[spec_dict["id"]] = {
        "id": spec_dict["id"],
        "displayName": spec_dict["displayName"],
        "hfModelId": spec_dict["hfModelId"],
        "template": spec_dict["template"],
        "family": spec_dict.get("family", "Custom"),
        "paramsB": spec_dict["paramsB"],
        "defaultCutoffLen": spec_dict["defaultCutoffLen"],
        "suggestedInstance": spec_dict["suggestedInstance"],
        "gated": spec_dict.get("gated", False),
        "notes": spec_dict.get("notes", "Auto-onboarded from Hugging Face."),
        "smokeTested": spec_dict.get("smokeTested", False),
        "imageTag": spec_dict.get("imageTag", "stable"),
        "serverlessModelId": serverless_id,
    }
    get_store().write_root_json(_CUSTOM_FILE, cur)


def delete_custom_model(model_id: str) -> bool:
    cur = _custom()
    if model_id not in cur:
        return False
    del cur[model_id]
    get_store().write_root_json(_CUSTOM_FILE, cur)
    return True
