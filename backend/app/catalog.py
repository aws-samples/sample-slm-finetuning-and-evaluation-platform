# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Engine-neutral model catalog (the manifest).

This is the spine of the platform. Each entry describes a model in
engine-NEUTRAL terms — model id, chat template, LoRA targets, sequence length,
suggested instance, tier — NOT LLaMA-Factory YAML. The renderer (render.py) is
the ONLY engine-aware code; it turns a manifest entry + hyperparams + a split
into LLaMA-Factory train/export YAML. Keeping the manifest neutral is cheap
insurance if the engine ever changes; per-model support is a data entry here,
never new code.

Tiers:
  - tiny   : 1–3B, cheap — used for smoke tests and fast iteration.
  - target : 4–20B, the prototype comparison targets.

`template` and `lora_target` values are taken from the live LLaMA-Factory repo
(README supported-models table + examples/train_lora/*.yaml). `instance` is a
suggested SageMaker training instance; it lives here so
instance choice is data, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Friendly provider names keyed by the Hugging Face org prefix (the part before
# the "/" in hf_model_id). The org IS the provider; this map just prettifies it
# for grouping in the UI. Unknown orgs fall back to a title-cased org name.
_PROVIDER_NAMES: dict[str, str] = {
    "qwen": "Alibaba (Qwen)",
    "microsoft": "Microsoft",
    "meta-llama": "Meta",
    "mistralai": "Mistral AI",
    "google": "Google",
    "openbmb": "OpenBMB (MiniCPM)",
    "thudm": "Zhipu / THUDM",
    "zai-org": "Zhipu / THUDM",
    "internlm": "InternLM",
    "deepseek-ai": "DeepSeek",
    "ibm-granite": "IBM Granite",
    "openai": "OpenAI",
    "tiiuae": "TII (Falcon)",
    "liquidai": "LiquidAI",
    "nvidia": "NVIDIA",
    "allenai": "Allen AI (OLMo)",
}


def provider_for(hf_model_id: str) -> str:
    """Friendly provider name from the HF org prefix (org/name → provider)."""
    org = hf_model_id.split("/", 1)[0] if "/" in hf_model_id else hf_model_id
    return _PROVIDER_NAMES.get(org.lower(), org.replace("-", " ").title())


# Placeholder for a model whose weight license we have NOT verified — used for
# custom/onboarded models (onboard.py builds a ModelSpec without a license) and
# for any catalog row we would otherwise have to guess at. It reads as a prompt to
# go look, which is the honest answer; asserting a wrong license is worse than
# asserting none.
LICENSE_UNKNOWN = "see model card"


@dataclass(frozen=True)
class ModelSpec:
    id: str  # catalog key used by the API + UI
    display_name: str
    hf_model_id: str  # model_name_or_path for LLaMA-Factory
    template: str  # LLaMA-Factory chat template name (verified vs live repo README)
    tier: str  # "tiny" | "target"
    params_b: float  # parameter count in billions (for the size hint)
    default_cutoff_len: int  # default sequence length
    suggested_instance: str  # SageMaker instance type hint
    family: str  # model family (Qwen3, Phi, GLM, …) — for grouping in the UI
    gated: bool = False  # requires accepting an HF license (needs HF token to train)
    # The base model's WEIGHT license, as an SPDX id where one applies (Apache-2.0,
    # MIT) or the publisher's own license name where it doesn't (Llama Community
    # License, Gemma Terms of Use). ADVISORY and independent of `gated`: a gated
    # repo can carry a permissive license (Mistral-7B is Apache-2.0 behind an HF
    # click-through), and an ungated repo can carry a restrictive custom license.
    # Whoever fine-tunes and redistributes is bound by this, not by `gated`.
    # Anything we could not pin down is left at the default, which says so rather
    # than guessing — the model card on Hugging Face is authoritative in all cases.
    license: str = LICENSE_UNKNOWN
    lora_target: str = "all"  # LLaMA-Factory LoRA target spec
    # Whether transformers/vLLM may execute Python that ships INSIDE the model repo
    # (a `config.json` `auto_map` pointing at custom modeling code). That code runs
    # with no sandbox — in the training job, in the eval container, and in any
    # endpoint created from the export bundle — under this deployment's execution
    # role. Default OFF: a model onboarded from an arbitrary Hugging Face repo id
    # must not get arbitrary code execution just by being onboarded. The curated
    # CATALOG rows below opt IN via _spec(), because those repos are known and
    # several of them (MiniCPM, GLM, InternLM, Phi) cannot load without it. To
    # onboard a custom model that needs it, set "trustRemoteCode": true on the
    # stored record deliberately.
    trust_remote_code: bool = False
    notes: str = ""
    # Which Docker image TIER this model trains/evals on (see aws_config
    # IMAGE_TIER_TAGS). "stable" = the proven 0.9.4 image the existing catalog
    # runs on; "latest" = the newer stack (0.9.5) for models 0.9.4 can't load.
    # The orchestrator resolves this tier → a concrete ECR image at launch, so
    # old models stay on the old image while new models get the newer one.
    image_tag: str = "stable"
    # Parameterization methods this model is expected to support (advisory hint
    # for the UI — like `gated`, it shapes the picker but the run is the source
    # of truth; verify-before-trust still applies per (model, method)). QLoRA
    # works wherever LoRA does (it only quantizes the base), so every catalog
    # model defaults to both. Narrow this per-model if a method proves unusable.
    allowed_methods: tuple[str, ...] = ("lora", "qlora")
    # Which training ENGINE this model's DEFAULT runs on. "llama_factory" is the
    # frozen-image SageMaker-training-job path every existing model uses. This is
    # advisory: the actual engine is a per-run choice (Hyperparams.engine), so the
    # SAME model can race on both engines as distinct leaderboard rows. Kept as a
    # field so the UI knows which engines a model offers.
    engine: str = "llama_factory"
    # SageMaker Public Hub model id for the serverless engine (e.g.
    # "huggingface-reasoning-qwen3-4b"). Empty ⇒ this model has no serverless
    # equivalent, so the serverless engine is not offered for it. Set only on rows
    # whose Public Hub equivalent has been confirmed to exist.
    serverless_model_id: str = ""

    @property
    def reasoning(self) -> bool:
        """True for families that emit a <think>/CoT prefix before the answer
        (Qwen3, DeepSeek-R1-Distill, GLM-Z1, gpt-oss). These need a larger eval
        token budget so the reasoning block CLOSES — an unclosed <think> gets
        stripped wholesale by eval.extract_answer, zeroing an otherwise-correct
        answer. Drives the reasoning-aware eval max_new_tokens floor. Detected
        from family/template so no per-spec annotation is needed; kept in sync
        with eval._REASONING_BLOCK_RES / profiler.detect_scaffold."""
        sig = f"{self.family} {self.template}".lower()
        return any(k in sig for k in ("qwen3", "deepseekr1", "deepseek-r1",
                                      "glmz1", "glm-z1", "gpt_oss", "gpt-oss"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "hfModelId": self.hf_model_id,
            "provider": provider_for(self.hf_model_id),
            "template": self.template,
            "tier": self.tier,
            "paramsB": self.params_b,
            "defaultCutoffLen": self.default_cutoff_len,
            "suggestedInstance": self.suggested_instance,
            "family": self.family,
            "gated": self.gated,
            "license": self.license,
            "loraTarget": self.lora_target,
            "trustRemoteCode": self.trust_remote_code,
            "notes": self.notes,
            "imageTag": self.image_tag,
            "allowedMethods": list(self.allowed_methods),
            "engine": self.engine,
            "reasoning": self.reasoning,
            "serverlessModelId": self.serverless_model_id,
            # Engines this model can run on: always llama_factory; + serverless
            # when it has a Public Hub mapping. Drives the UI engine picker.
            "engines": (["llama_factory", "sagemaker_serverless"]
                        if self.serverless_model_id else ["llama_factory"]),
        }


# Catalog. Both image tiers.
# Curated catalog: ungated, text-only SLMs ≤ ~10B with templates verified
# against the live LLaMA-Factory README (commit a98a1ef). Tiers are size bands:
#   tiny  ≤ 2B   ·  small 3–4B  ·  mid 7–10B  ·  large > 10B
# A couple of gated families are kept but flagged (need an HF token to train).
# Full-weight (non-adapter) methods: they update/keep the WHOLE model in the
# optimizer, so they need far more GPU memory than LoRA/QLoRA and produce a
# standalone model (no adapter to merge). Kept as a module constant so render.py,
# limits.py and the instance picker all agree on what "heavyweight" means.
FULL_WEIGHT_METHODS = ("full", "freeze")


def _instance_for(params_b: float, method: str = "lora") -> str:
    """Cheapest training-appropriate instance that FITS the model for `method`.

    Sizing rationale (verified empirically — see the catalog smoke-test campaign):
    - ≤8B fits a single A10G (24 GB) → g5, the cheapest training-grade card (the
      L4-based g6 is cheaper/hr but has HALF the memory bandwidth, so it trains
      SLOWER — wrong tool for training).
    - 9–14B OOMs on g5: a 24 GB A10G can't hold the fp16 weights+activations, and
      the multi-GPU g5.12xlarge runs naive DDP (a FULL model copy per 24 GB GPU,
      so the 4×24 GB doesn't pool → still OOM). A single L40S (48 GB, g6e) holds a
      14B on ONE GPU (no DDP) AND is cheaper than the g5.12xlarge that fails.
    - >14B needs the 4×48 GB g6e.12xlarge.
    QLoRA fits a tier smaller, but we size for the heavier fp16-LoRA case so either
    method works on the suggested instance.

    FULL/FREEZE (full-weight) is the heavy case: AdamW keeps weights+grads+optimizer
    state for EVERY parameter (~16–20 B/param), so even a 2B model needs ~34 GB
    (PROVEN on a real g6e.2xlarge probe 2026-06-19: qwen3-1.7b full-FT peaked at
    76% of 45 GiB = ~34 GB). A 24 GB A10G (g5) CANNOT hold it. So full-weight
    methods route to the L40S (48 GB) g6e.2xlarge regardless of the (≤2B) size.
    Models >2B are not offered full/freeze in Phase 1 (see _spec.allowed_methods),
    so this single band suffices; g7e (96 GB) is deferred until a Blackwell-ready
    image exists (the 0.9.4/0.9.5 torch has no sm_120 kernels — proven blocker).
    """
    if method in FULL_WEIGHT_METHODS:
        return "ml.g6e.2xlarge"     # 1× L40S 48 GB — the proven single-GPU full-FT card
    if params_b <= 2:
        return "ml.g5.2xlarge"      # 1× A10G 24 GB
    if params_b <= 4:
        return "ml.g5.4xlarge"      # 1× A10G 24 GB
    if params_b <= 8:
        return "ml.g5.8xlarge"      # 1× A10G 24 GB
    if params_b <= 14:
        return "ml.g6e.2xlarge"     # 1× L40S 48 GB — fits 9–14B on one GPU (no DDP OOM)
    return "ml.g6e.12xlarge"        # 4× L40S 192 GB — >14B


# Phase-1 single-GPU full/freeze ceiling. A real g6e.2xlarge probe (2026-06-19)
# MEASURED qwen3-1.7b full-FT at ~34 GB / 45 GiB (≈20 GB/param). Extrapolating,
# 3B ≈ ~60 GB > 45 GiB → OOM. (The 3B confirmation probe was blocked by g6e
# capacity unavailability in us-east-1, but the measured 20 GB/param anchor is
# decisive — 3B does not fit.) So full/freeze is offered only for models AT OR
# BELOW this size. To lift it: an 8-bit/paged optimizer (~10 GB/param) or the
# g7e/Blackwell 96 GB image. Bumping this constant is the single knob to widen it.
FULL_FREEZE_MAX_PARAMS_B = 2.0


def _spec(id, name, hf, template, family, params_b, *, gated=False, cutoff=2048,
          notes="", image_tag="stable", serverless_model_id="",
          license=LICENSE_UNKNOWN, trust_remote_code=True):
    band = "tiny" if params_b <= 2 else "small" if params_b <= 4 else "mid" if params_b <= 10 else "large"
    # LoRA/QLoRA are offered for every model. full/freeze (full-weight) are offered
    # ONLY for small models that fit one g6e GPU (≤FULL_FREEZE_MAX_PARAMS_B) — they
    # produce a standalone model (no adapter) and need ~16–20 B/param of GPU memory.
    # This is the per-model size gate; the (engine, stage, method) matrix in
    # Hyperparams further restricts full/freeze to the llama_factory engine + SFT.
    methods = ["lora", "qlora"]
    if params_b <= FULL_FREEZE_MAX_PARAMS_B:
        methods += ["full", "freeze"]
    return ModelSpec(
        id=id, display_name=name, hf_model_id=hf, template=template, family=family,
        tier=band, params_b=params_b, default_cutoff_len=cutoff,
        suggested_instance=_instance_for(params_b), gated=gated, notes=notes,
        image_tag=image_tag, serverless_model_id=serverless_model_id,
        allowed_methods=tuple(methods), license=license,
        # Curated rows only. These are named, reviewed repos, and several of them
        # (MiniCPM4, GLM-4, GLM-Z1, InternLM2.5, Phi-3.5/4-mini, LFM2) ship their
        # architecture as repo-side modeling code and will not load without it.
        # ModelSpec defaults this to False so user-onboarded repo ids do NOT
        # inherit it — see the field comment on ModelSpec.trust_remote_code.
        trust_remote_code=trust_remote_code,
    )


# LICENSES: every row carries the base model's weight license (see
# ModelSpec.license). SPDX ids where the publisher used a standard license; the
# publisher's own license name where they didn't; LICENSE_UNKNOWN ("see model
# card") where the terms are custom enough — or have changed enough across the
# family's releases — that restating them here would risk being wrong. Re-verify
# against the Hugging Face model card before relying on any of these; publishers
# do relicense (microsoft/phi-4 moved to MIT after its initial release).
CATALOG: list[ModelSpec] = [
    # --- Qwen3 (ungated) --- Apache-2.0 across the Qwen3 dense releases.
    _spec("qwen3-0.6b", "Qwen3 0.6B", "Qwen/Qwen3-0.6B", "qwen3", "Qwen3", 0.6,
          license="Apache-2.0",
          serverless_model_id="huggingface-reasoning-qwen3-06b"),
    _spec("qwen3-1.7b", "Qwen3 1.7B", "Qwen/Qwen3-1.7B", "qwen3", "Qwen3", 1.7,
          license="Apache-2.0",
          serverless_model_id="huggingface-reasoning-qwen3-1-7b"),
    # Instruct-2507 is the NON-thinking Qwen3 variant (plain ChatML, no <think>), so
    # it uses qwen3_nothink — NOT the thinking "qwen3" template, which would inject
    # empty <think></think> tokens with loss during SFT. (qwen3_nothink ships in both
    # the 0.9.4 and 0.9.5 images — verified against the capability manifests.)
    _spec("qwen3-4b", "Qwen3 4B Instruct 2507", "Qwen/Qwen3-4B-Instruct-2507", "qwen3_nothink", "Qwen3", 4.0,
          license="Apache-2.0",
          notes="Proven prototype that beat Sonnet 4.5 on the narrow task.",
          serverless_model_id="huggingface-reasoning-qwen3-4b"),
    _spec("qwen3-8b", "Qwen3 8B", "Qwen/Qwen3-8B", "qwen3", "Qwen3", 8.0,
          license="Apache-2.0",
          serverless_model_id="huggingface-reasoning-qwen3-8b"),
    # --- Qwen2.5 (ungated) --- mostly Apache-2.0, but NOT uniformly: the 3B and
    # 72B sizes shipped under Qwen's own research/community terms instead, so the
    # 3B row below is deliberately not marked Apache-2.0.
    _spec("qwen2.5-0.5b", "Qwen2.5 0.5B Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "qwen", "Qwen2.5", 0.5,
          license="Apache-2.0"),
    _spec("qwen2.5-1.5b", "Qwen2.5 1.5B Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "qwen", "Qwen2.5", 1.5,
          license="Apache-2.0"),
    _spec("qwen2.5-3b", "Qwen2.5 3B Instruct", "Qwen/Qwen2.5-3B-Instruct", "qwen", "Qwen2.5", 3.0,
          license="Qwen Research License — see model card",
          notes="Unlike the other Qwen2.5 sizes this one is NOT Apache-2.0; check its "
                "license terms before commercial use or redistribution."),
    _spec("qwen2.5-7b", "Qwen2.5 7B Instruct", "Qwen/Qwen2.5-7B-Instruct", "qwen", "Qwen2.5", 7.0,
          license="Apache-2.0",
          serverless_model_id="huggingface-llm-qwen2-5-7b-instruct"),
    # --- Phi (ungated, MIT) ---
    _spec("phi-3.5-mini", "Phi-3.5-mini Instruct", "microsoft/Phi-3.5-mini-instruct", "phi", "Phi", 3.8,
          license="MIT"),
    # Phi-4-mini needs the NEWER stack: on 0.9.4 it fails at import
    # (`ImportError: cannot import name 'LossKwargs'`) because its trust_remote_code
    # modeling requires transformers v5. LLaMA-Factory 0.9.5 ships transformers v5
    # AND a dedicated `phi4_mini` template — so this model runs on the `latest`
    # image tier, not `stable`. This is the canonical case the multi-image design
    # exists for: a new model that the frozen image can't load.
    _spec("phi-4-mini", "Phi-4-mini Instruct", "microsoft/Phi-4-mini-instruct", "phi4_mini", "Phi", 3.8,
          image_tag="latest", license="MIT",
          notes="Needs transformers v5 (LLaMA-Factory 0.9.5) — runs on the 'latest' image tier."),
    # --- MiniCPM4 (ungated) --- OpenBMB has shipped MiniCPM under several different
    # terms across generations (a custom General Model License on early releases,
    # Apache-2.0 on later ones), so these are left unasserted.
    _spec("minicpm4-0.5b", "MiniCPM4 0.5B", "openbmb/MiniCPM4-0.5B", "cpm4", "MiniCPM", 0.5,
          license="OpenBMB MiniCPM terms — see model card"),
    _spec("minicpm4-8b", "MiniCPM4 8B", "openbmb/MiniCPM4-8B", "cpm4", "MiniCPM", 8.0,
          license="OpenBMB MiniCPM terms — see model card"),
    # --- GLM-4 (ungated) --- custom THUDM license, not an OSI license.
    _spec("glm-4-9b", "GLM-4 9B Chat", "THUDM/glm-4-9b-chat", "glm4", "GLM", 9.0,
          license="GLM-4 Model License — see model card"),
    # --- InternLM2.5 (ungated) --- code is Apache-2.0; the WEIGHTS carry separate
    # InternLM terms (free for research, commercial use subject to their form), so
    # the weight license is the one that matters here and it isn't plain Apache-2.0.
    _spec("internlm2.5-7b", "InternLM2.5 7B Chat", "internlm/internlm2_5-7b-chat", "intern2", "InternLM", 7.0,
          license="InternLM model-weight terms — see model card"),
    # --- DeepSeek-R1-Distill (ungated; reasoning distillations) --- DeepSeek
    # released the R1 distills under MIT; note the Qwen-derived ones also inherit
    # obligations from their Qwen2.5 base.
    _spec("deepseek-r1-distill-qwen-1.5b", "DeepSeek-R1-Distill-Qwen 1.5B",
          "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseekr1", "DeepSeek-R1-Distill", 1.5,
          license="MIT (distilled from a Qwen2.5 base)",
          serverless_model_id="deepseek-llm-r1-distill-qwen-1-5b"),
    _spec("deepseek-r1-distill-qwen-7b", "DeepSeek-R1-Distill-Qwen 7B",
          "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseekr1", "DeepSeek-R1-Distill", 7.0,
          license="MIT (distilled from a Qwen2.5 base)",
          serverless_model_id="deepseek-llm-r1-distill-qwen-7b"),
    # --- IBM Granite 3 (ungated, Apache-2.0) ---
    _spec("granite-3.1-2b", "Granite 3.1 2B Instruct", "ibm-granite/granite-3.1-2b-instruct", "granite3", "Granite", 2.0,
          license="Apache-2.0"),
    _spec("granite-3.1-8b", "Granite 3.1 8B Instruct", "ibm-granite/granite-3.1-8b-instruct", "granite3", "Granite", 8.0,
          license="Apache-2.0"),
    # --- GPT-OSS (ungated MoE, >10B; QLoRA likely needed) ---
    _spec("gpt-oss-20b", "GPT-OSS 20B", "openai/gpt-oss-20b", "gpt_oss", "GPT-OSS", 20.0, cutoff=4096,
          license="Apache-2.0",
          notes="MoE; largest option. Heavier instance + QLoRA likely needed.",
          serverless_model_id="openai-reasoning-gpt-oss-20b"),
    # --- larger ungated additions (templates verified vs live LLaMA-Factory
    #     README supported-models table, commit 053d43c, 2026-06-03) ---
    _spec("qwen3-14b", "Qwen3 14B", "Qwen/Qwen3-14B", "qwen3", "Qwen3", 14.0,
          license="Apache-2.0",
          serverless_model_id="huggingface-reasoning-qwen3-14b"),
    _spec("qwen2.5-14b", "Qwen2.5 14B Instruct", "Qwen/Qwen2.5-14B-Instruct", "qwen", "Qwen2.5", 14.0,
          license="Apache-2.0",
          serverless_model_id="huggingface-llm-qwen2-5-14b-instruct"),
    _spec("phi-4", "Phi-4 (14B)", "microsoft/phi-4", "phi4", "Phi", 14.0,
          license="MIT",
          notes="Full Phi-4; template phi4 (distinct from phi4_mini)."),
    _spec("deepseek-r1-distill-qwen-14b", "DeepSeek-R1-Distill-Qwen 14B",
          "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseekr1", "DeepSeek-R1-Distill", 14.0,
          license="MIT (distilled from a Qwen2.5 base)",
          serverless_model_id="deepseek-llm-r1-distill-qwen-14b"),
    # The GLM-*-0414 series was published under different terms than the older
    # glm-4-9b-chat above; not restated here because GLM licensing has moved
    # release to release.
    _spec("glm-z1-9b", "GLM-Z1 9B (reasoning)", "zai-org/GLM-Z1-9B-0414", "glmz1", "GLM", 9.0,
          license=LICENSE_UNKNOWN,
          notes="GLM reasoning variant; template glmz1."),
    # Falcon-H1 (hybrid attention+Mamba/SSM) REMOVED: the architecture loads +
    # trains, but LLaMA-Factory's LoRA merge/export can't handle the hybrid-Mamba
    # layers — every run produced a 0-byte merged artifact and failed at export
    # (verified on a real race 2026-06-10). Re-add only if LF gains hybrid-Mamba
    # merge support, or behind an adapter-only (no-merge) deploy path.
    # --- LiquidAI LFM2 (ungated) — NEEDS THE `latest` IMAGE TIER ---
    # LFM2-MoE's architecture (model_type=lfm2_moe, Lfm2MoeForCausalLM) + the
    # `lfm2` chat template only exist in LLaMA-Factory 0.9.5 / transformers v5.
    # On the `stable` 0.9.4 image, transformers rejects it ("model type
    # `lfm2_moe` ... Transformers does not recognize this architecture ... your
    # version of Transformers is out of date"). This is the canonical multi-image
    # case: a new model release that trains on `latest` but not `stable`.
    _spec("lfm2-8b-a1b", "LFM2-8B-A1B (MoE)", "LiquidAI/LFM2-8B-A1B", "lfm2", "LFM2", 8.0,
          image_tag="latest", license="LFM Open License — see model card",
          notes="LiquidAI LFM2 MoE. Needs LLaMA-Factory 0.9.5 (transformers v5) — runs on the 'latest' image tier."),
    # --- gated families (kept for reference; need an HF token to train) ---
    # `gated` and `license` are separate facts. The Llama and Gemma rows are gated
    # AND carry the publisher's own non-OSI license; mistral-7b is gated on Hugging
    # Face but the weights themselves are Apache-2.0.
    _spec("llama-3.2-1b", "Llama 3.2 1B Instruct", "meta-llama/Llama-3.2-1B-Instruct", "llama3", "Llama", 1.0, gated=True,
          license="Llama 3.2 Community License",
          serverless_model_id="meta-textgeneration-llama-3-2-1b-instruct"),
    _spec("llama-3.2-3b", "Llama 3.2 3B Instruct", "meta-llama/Llama-3.2-3B-Instruct", "llama3", "Llama", 3.0, gated=True,
          license="Llama 3.2 Community License",
          serverless_model_id="meta-textgeneration-llama-3-2-3b-instruct"),
    _spec("llama-3.1-8b", "Llama 3.1 8B Instruct", "meta-llama/Llama-3.1-8B-Instruct", "llama3", "Llama", 8.0, gated=True,
          license="Llama 3.1 Community License",
          serverless_model_id="meta-textgeneration-llama-3-1-8b-instruct"),
    _spec("mistral-7b", "Mistral 7B Instruct v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "mistral", "Mistral", 7.0, gated=True,
          license="Apache-2.0",
          notes="Apache-2.0 weights behind a gated Hugging Face repo — the gate is an "
                "access click-through, not a restrictive weight license."),
    _spec("gemma-2-9b", "Gemma 2 9B Instruct", "google/gemma-2-9b-it", "gemma2", "Gemma", 9.0, gated=True,
          license="Gemma Terms of Use"),
]

_BY_ID = {m.id: m for m in CATALOG}


def _apply_serverless_overlay(spec: "ModelSpec") -> "ModelSpec":
    """Fill an EMPTY serverless_model_id from the runtime config.json overlay (a
    tag auto-discovered + applied via serverless_catalog.register_serverless_model_id),
    returning a NEW ModelSpec. A hand-curated static tag is the floor and is NEVER
    overridden — the overlay only fills empties. Returns the spec unchanged on any
    error or when no overlay entry applies. Produces a ModelSpec (not just a dict)
    so every launch/verification gate — which reads serverless_model_id off the
    spec — honors a discovered tag uniformly."""
    if spec.serverless_model_id:
        return spec  # static tag wins
    try:
        from dataclasses import replace

        from .serverless_catalog import serverless_overlay

        hub_id = serverless_overlay().get(spec.id)
        if hub_id:
            return replace(spec, serverless_model_id=hub_id)
    except Exception:  # noqa: BLE001 — overlay is advisory, never break the catalog
        pass
    return spec


def list_models() -> list[dict[str, Any]]:
    """Built-in catalog + any auto-onboarded custom models (persisted), each with
    the runtime serverless-tag overlay applied (fills empty serverless ids)."""
    out = [_apply_serverless_overlay(m).to_dict() for m in CATALOG]
    try:
        from .onboard import get_custom_model, list_custom_models

        # Always surface every persisted custom model as a REMOVABLE row, even
        # when its slug collides with a built-in id. Resolve from the custom
        # record directly (get_model would return the built-in spec on a
        # collision, hiding the custom row's Remove button — the NO-REMOVE
        # asymmetry). Mark custom:True so the UI shows Remove regardless.
        by_id = {m["id"]: i for i, m in enumerate(out)}
        for d in list_custom_models():
            spec = get_custom_model(d["id"])
            if not spec:
                continue
            row = _apply_serverless_overlay(spec).to_dict()
            row["custom"] = True
            if d["id"] in by_id:
                out[by_id[d["id"]]] = row  # custom record wins on a collision
            else:
                out.append(row)
    except Exception:  # noqa: BLE001 — never let onboarding break the catalog
        pass
    return out


def get_model(model_id: str) -> ModelSpec | None:
    """Built-in model, else an auto-onboarded custom model (persisted), with the
    runtime serverless-tag overlay applied (so a discovered tag reaches the launch
    + verification gates that read serverless_model_id off the returned spec)."""
    spec = _BY_ID.get(model_id)
    if spec is None:
        try:
            from .onboard import get_custom_model

            spec = get_custom_model(model_id)
        except Exception:  # noqa: BLE001
            spec = None
    return _apply_serverless_overlay(spec) if spec is not None else None


def model_id_for_hf(hf_model_id: str) -> str | None:
    """Reverse lookup: a HuggingFace model id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')
    → the catalog id ('qwen2.5-0.5b'), or None if not in the catalog. Used so the
    base-model leaderboard row labels the SAME way as the fine-tuned row (which is
    derived from the catalog id), instead of from the raw HF repo name."""
    if not hf_model_id:
        return None
    for m in CATALOG:
        if m.hf_model_id == hf_model_id:
            return m.id
    try:
        from .onboard import list_custom_models

        for d in list_custom_models():
            if d.get("hfModelId") == hf_model_id:
                return d.get("id")
    except Exception:  # noqa: BLE001
        pass
    return None


# KTO per-class loss-weight bounds. 1.0 = neutral (LF default); the KTO paper's
# imbalance fix rarely needs more than ~4× on the minority class, and ≤0 would
# zero/destabilize a class, so clamp to [0.1, 4.0].
_KTO_WEIGHT_MIN = 0.1
_KTO_WEIGHT_MAX = 4.0

# LoRA+ B/A learning-rate ratio bounds. 1 = plain LoRA (no benefit); the paper's
# default is ~16 and very high ratios destabilize the B matrix. Clamp to [1, 128]
# (mirrors bounds()) so a typo or stale clone can't pass a degenerate ratio.
_LORAPLUS_RATIO_MIN = 1.0
_LORAPLUS_RATIO_MAX = 128.0

# SimPO's target reward margin (γ in the SimPO paper). LLaMA-Factory default 0.5;
# values near 0 collapse to a margin-free loss (SimPO degenerates to length-
# normalized reference-free DPO) and >~2 are extreme. Clamp to a POSITIVE band so a
# typo/stale clone can't silently pick "SimPO" yet train a margin-free loss — the
# floor is 0.1, not 0, so SimPO always means a real margin. Only used when pref_loss=simpo.
_SIMPO_GAMMA_MIN = 0.1
_SIMPO_GAMMA_MAX = 2.0

# NEFTune embedding-noise alpha (HF TrainingArguments passthrough). 0 = off; the
# paper's sweet spot is ~5 and very large values hurt. Clamp to [0, 15]. Applies to
# any stage (it's a training-time regularizer), emitted only when > 0.
_NEFTUNE_ALPHA_MIN = 0.0
_NEFTUNE_ALPHA_MAX = 15.0


@dataclass
class Hyperparams:
    """Editable training knobs surfaced in the UI.

    Defaults mirror the LLaMA-Factory example SFT config. `cutoff_len` defaults
    to the model's `default_cutoff_len` when not provided.
    """

    # Objective (LLaMA-Factory `stage`). "sft" is the proven default; "dpo" trains
    # on preference pairs (chosen/rejected); "kto" on independently-labelled
    # desirable/undesirable completions (binary good/bad — cheaper than DPO, no
    # pairing needed). Same engine, same merge/eval, only the loss differs. The
    # dataset must be the matching SHAPE (sft=messages, dpo=ranking, kto=labelled)
    # — guarded upstream, not here.
    stage: str = "sft"
    # Training ENGINE for THIS run. "llama_factory" (default) keeps the existing
    # path byte-identical. "sagemaker_serverless" routes to SageMaker's managed
    # serverless model-customization (SFT/DPO/RLVR). Per-run so the same model can
    # race on both engines as distinct leaderboard rows (mirrors lora/qlora).
    engine: str = "llama_factory"
    # Parameterization (LLaMA-Factory `finetuning_type`). "lora" is the proven
    # default that keeps the existing SFT-LoRA path byte-identical; "qlora" is
    # LoRA on a 4-bit-quantized base (set quantization_bit=4) — same adapter, same
    # merge/export path, just a smaller memory footprint so bigger models fit on
    # the same GPU. quantization_bit is None for plain LoRA and 4 for QLoRA; it is
    # NEVER emitted into the export/merge config (merging onto a quantized base is
    # unsupported — see render.py).
    finetuning_type: str = "lora"
    quantization_bit: int | None = None
    # DPO-only: the preference-loss beta (KL strength). Ignored unless stage=dpo.
    pref_beta: float = 0.1
    # Preference-loss FAMILY (LLaMA-Factory pref_loss). The objective stays
    # stage="dpo" for all three — only the loss differs, so they share the
    # preference (chosen/rejected) dataset shape:
    #   sigmoid → standard DPO (the default; uses a frozen reference model + beta)
    #   orpo    → ORPO (reference-free; folds preference into one SFT-style stage)
    #   simpo   → SimPO (reference-free; length-normalized reward + target margin)
    # ORPO/SimPO are REFERENCE-FREE (LF: use_ref_model is False when pref_loss is
    # orpo/simpo) — cheaper, no 2nd resident model. Default "sigmoid" so a plain DPO
    # run stays byte-identical. Ignored unless stage=dpo.
    pref_loss: str = "sigmoid"
    # SimPO-only: the target reward margin γ (LF simpo_gamma, default 0.5). Only
    # reaches the trainer for stage=dpo + pref_loss=simpo (render gates on it);
    # clamped to [0,2] in __post_init__. Ignored otherwise.
    simpo_gamma: float = 0.5
    # KTO-only: per-class loss weights (LLaMA-Factory kto_chosen_weight /
    # kto_rejected_weight; = the KTO paper's λ_D / λ_U). Default 1.0/1.0 = the
    # LF default, so an unchanged value emits NOTHING into the YAML and a KTO run
    # stays byte-identical to before. Raising the weight on the minority class is
    # the KTO paper's prescribed fix for label imbalance (Ethayarajh et al. 2024,
    # §4.2): keep (λD·nD)/(λU·nU) in [1, 4/3]. Ignored unless stage=kto.
    kto_chosen_weight: float = 1.0
    kto_rejected_weight: float = 1.0
    # RLVR-only (serverless engine): which VERIFIABLE reward drives GRPO. EITHER a
    # PRESET reward (gsm8k / prime_math / prime_code) OR a CUSTOM reward function
    # (reward_function_id → a user-authored reward Lambda registered as an
    # Evaluator; see reward_functions.py). Exactly one is set when stage=rlvr
    # (gated in __post_init__); both ignored otherwise.
    preset_reward_function: str = ""
    reward_function_id: str = ""
    # RLAIF-only (serverless engine): RL from AI Feedback. The reward is a NON-
    # verifiable AI judge driven by a reward PROMPT (a reward_function_id pointing
    # at a 'reward_prompt'-kind registry record holding the judge prompt text), not
    # a verifiable reward function/Lambda. `reward_model_id` names the judge model
    # the recipe scores rollouts with. Set only when stage=rlaif; cleared otherwise.
    reward_model_id: str = ""
    lora_rank: int = 8
    lora_alpha: int | None = None  # LLaMA-Factory defaults to 2*lora_rank if unset
    # LoRA VARIANT — a modifier ON TOP of finetuning_type=lora (NOT a new method).
    # All ride the existing LoRA path (rank/alpha/target + merge/export unchanged in
    # shape), so they're cheap to offer. "lora" = plain (byte-identical to before).
    #   dora     → use_dora=true        (weight-decomposed LoRA; merges differently)
    #   rslora   → use_rslora=true      (rank-stabilized scaling)
    #   pissa    → pissa_init=true + pissa_iter=16 (better SVD-based init)
    #   loraplus → loraplus_lr_ratio=16 (separate, higher LR for the B matrix)
    # Verified supported by LLaMA-Factory; each still needs a verify-before-trust
    # smoke test on OUR image (esp. DoRA, whose merge path differs). Only emitted
    # for adapter methods (lora/qlora); ignored for full/freeze.
    # CONSTRAINT: DoRA + PiSSA require the FULL-PRECISION weight matrix, so they
    # CANNOT run on a 4-bit QLoRA base (PEFT hard-rejects them at model-load) —
    # __post_init__ rejects quantization_bit + {dora,pissa}. rsLoRA/LoRA+ are fine
    # on QLoRA. See QUANT_INCOMPATIBLE_VARIANTS.
    lora_variant: str = "lora"
    loraplus_lr_ratio: float = 16.0  # LoRA+ B/A learning-rate ratio (paper default)
    # freeze-only: how many of the top transformer layers to train (LLaMA-Factory
    # freeze_trainable_layers; positive = last N blocks). Ignored unless
    # finetuning_type=="freeze". Default 2 = a light, low-memory "step up from LoRA".
    freeze_trainable_layers: int = 2
    learning_rate: float = 1.0e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    cutoff_len: int | None = None
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    save_steps: int = 500
    max_samples: int | None = None  # cap rows for quick runs; None = all
    # Early stopping. Requires the dataset to have a validation split
    # (the in-training eval signal). When enabled, num_train_epochs becomes a
    # CEILING ("max epochs") and training stops after `patience` evals with no
    # eval_loss improvement; the EXPORTED model is the best checkpoint, not the
    # last (load_best_model_at_end). Ignored if the dataset has no val set.
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 3
    # --- Efficiency knobs (orthogonal to objective/method; LLaMA-Factory engine
    # only — the serverless managed recipe doesn't expose them). Each is emitted
    # into the train YAML ONLY when non-default, so an unchanged run stays
    # byte-identical to before these existed.
    #   neftune_noise_alpha → HF TrainingArguments passthrough: adds uniform noise
    #     to embeddings during training, a frequently-free quality bump. 0 = off
    #     (default); paper sweet spot ~5. Clamped to [0,15]. Applies to ANY stage.
    #   enable_liger_kernel → LLaMA-Factory model_arg: fused Liger kernels for less
    #     memory + faster training. Drop-in; version-sensitive, so verify-before-trust
    #     still applies (a run that crashes marks the model untested, not the image).
    #   packing → concatenate short samples to one cutoff_len sequence (throughput
    #     win on short SFT data). SFT-ONLY (it changes how examples are batched, which
    #     is unsafe for preference/KTO loss). neat_packing additionally masks
    #     cross-sample attention (needs FlashAttention) — we expose plain `packing`
    #     only and leave neat_packing off to avoid the FA2 dependency on our image.
    neftune_noise_alpha: float = 0.0
    enable_liger_kernel: bool = False
    packing: bool = False

    # Methods supported. "lora" keeps the original path; "qlora" adds
    # quantization_bit=4. "full"/"freeze" are FULL-WEIGHT methods (no adapter,
    # standalone model) — added in the FFT phase. They are further gated to the
    # llama_factory engine + SFT + small models (see ENGINE_METHODS,
    # FULL_FREEZE_STAGES and the per-model allowed_methods size gate).
    SUPPORTED_FINETUNING_TYPES = ("lora", "qlora", "full", "freeze")
    # LoRA adapter variants (modifiers on lora/qlora — see lora_variant field).
    SUPPORTED_LORA_VARIANTS = ("lora", "dora", "rslora", "pissa", "loraplus")
    # Preference-loss families for stage=dpo (see pref_loss field). All ride the
    # preference (chosen/rejected) dataset; only the loss differs. "sigmoid" = plain
    # DPO (default). orpo/simpo are REFERENCE-FREE. (LLaMA-Factory also offers
    # hinge/ipo/kto_pair, not exposed here.)
    SUPPORTED_PREF_LOSS = ("sigmoid", "orpo", "simpo")
    # The reference-free preference losses — no frozen reference model needed
    # (cheaper). LF derives use_ref_model = (stage==dpo and pref_loss not in these).
    REFERENCE_FREE_PREF_LOSS = ("orpo", "simpo")
    # Variants that REQUIRE the full-precision weight matrix and so CANNOT run on a
    # 4-bit (QLoRA) base: DoRA decomposes weights into magnitude×direction, PiSSA
    # initializes from an SVD of the weights — PEFT/LLaMA-Factory hard-reject both on
    # a PTQ-quantized base ("DoRA is not compatible with PTQ-quantized models"; PiSSA
    # demands scripts/pissa_init.py for a quantized model). rsLoRA (scaling tweak) and
    # LoRA+ (per-matrix LR) need no full-precision weights, so they're fine on QLoRA.
    # This is the source of truth for the UI gate + the __post_init__ guard so an
    # invalid combo can't reach a billable launch (cost a real race 2 failed jobs).
    QUANT_INCOMPATIBLE_VARIANTS = ("dora", "pissa")
    # Stages full-weight (full/freeze) training is allowed for in Phase 1. SFT only
    # — DPO/KTO full-FT need a resident reference model (more memory, smaller models)
    # and RLVR/RLAIF are serverless-only GRPO. Adapter methods keep the full matrix.
    FULL_FREEZE_STAGES = ("sft",)
    # THE full-FT learning-rate gotcha (correctness-critical). LoRA's default LR is
    # 1e-4; full-weight training at that LR DIVERGES / catastrophically forgets.
    # LLaMA-Factory's own full_sft example uses 1e-5. We (a) default full/freeze to
    # 1e-5 in the UI, and (b) CLAMP it in __post_init__ as a backend safety net that
    # can't be bypassed: any full/freeze LR above the ceiling is snapped to the
    # default (the common failure is silently inheriting LoRA's 1e-4).
    FULL_WEIGHT_DEFAULT_LR = 1.0e-5
    FULL_WEIGHT_MAX_LR = 5.0e-5
    # Objectives (LLaMA-Factory `stage`) supported so far. sft=messages data,
    # dpo=preference (ranking) data, kto=binary-labelled completions, rlvr=GRPO
    # against a verifiable reward (serverless engine only — AWS's managed VERL
    # stack; our LLaMA-Factory image can't do it).
    SUPPORTED_STAGES = ("sft", "dpo", "kto", "rlvr", "rlaif")
    # Training engines a run can target.
    SUPPORTED_ENGINES = ("llama_factory", "sagemaker_serverless")
    # Per-engine capability matrix — which (stage, method) cells each engine can
    # fill. The serverless engine does SFT/DPO/RLVR with LoRA only — NO kto, NO
    # qlora (serverless offers LoRA/FULL, not 4-bit; kto has no serverless
    # recipe). RLVR is serverless-ONLY (LLaMA-Factory can't).
    # llama_factory keeps its full non-RLVR matrix. Used to gate combinations
    # honestly in the UI + reject bad requests early.
    ENGINE_STAGES = {
        "llama_factory": ("sft", "dpo", "kto"),
        "sagemaker_serverless": ("sft", "dpo", "rlvr", "rlaif"),
    }
    # full/freeze run on the llama_factory engine only. Serverless stays lora-only:
    # the live SageMaker Public Hub exposes NO full-parameter recipe for any
    # open-weight model we serve (only Amazon Nova carries bare full recipes, and
    # Nova has no public HF repo / isn't in our catalog) — so serverless full/freeze
    # is not buildable today (verified on the live hub 2026-06-19).
    ENGINE_METHODS = {
        "llama_factory": ("lora", "qlora", "full", "freeze"),
        "sagemaker_serverless": ("lora",),
    }
    # RLVR "preset" rewards. AWS REMOVED preset_reward_function from the open-weight
    # GRPO recipe (the RLVRTrainer takes ONLY a custom-reward Evaluator ARN), so a
    # preset is now reconstructed as an auto-provisioned built-in reward FUNCTION:
    # gsm8k/prime_math → the verifiable numeric_match scorer, resolved to an Evaluator
    # ARN at launch (see reward_functions.PRESET_BUILTIN_REWARDS). prime_code was
    # DROPPED — a pure-python reward Lambda can't run code against tests, so a
    # built-in would mislead; author a custom reward with execution tooling instead.
    PRESET_REWARD_FUNCTIONS = ("gsm8k", "prime_math")

    def __post_init__(self) -> None:
        # Normalize the (finetuning_type, quantization_bit) pair so callers can
        # send just the method and never an inconsistent combination:
        #   qlora → force a 4-bit base (the only thing that makes it QLoRA)
        #   lora  → no quantization (a quantization_bit here would be ignored;
        #           clear it so the rendered config can't drift)
        if self.finetuning_type not in self.SUPPORTED_FINETUNING_TYPES:
            raise ValueError(
                f"unsupported finetuning_type {self.finetuning_type!r}; "
                f"expected one of {self.SUPPORTED_FINETUNING_TYPES}"
            )
        if self.stage not in self.SUPPORTED_STAGES:
            raise ValueError(
                f"unsupported stage {self.stage!r}; expected one of {self.SUPPORTED_STAGES}"
            )
        if self.engine not in self.SUPPORTED_ENGINES:
            raise ValueError(
                f"unsupported engine {self.engine!r}; expected one of {self.SUPPORTED_ENGINES}"
            )
        # Per-engine capability gating: a run can only ask for a (stage, method)
        # the chosen engine actually supports. Catches e.g. serverless+qlora or
        # serverless+kto before a billable launch. llama_factory keeps its full
        # matrix, so existing runs are unaffected.
        if self.stage not in self.ENGINE_STAGES.get(self.engine, self.SUPPORTED_STAGES):
            raise ValueError(
                f"engine {self.engine!r} does not support stage {self.stage!r}; "
                f"supported: {self.ENGINE_STAGES.get(self.engine)}"
            )
        if self.finetuning_type not in self.ENGINE_METHODS.get(self.engine, self.SUPPORTED_FINETUNING_TYPES):
            raise ValueError(
                f"engine {self.engine!r} does not support method {self.finetuning_type!r}; "
                f"supported: {self.ENGINE_METHODS.get(self.engine)}"
            )
        if self.finetuning_type == "qlora":
            if self.quantization_bit is None:
                self.quantization_bit = 4
        else:
            self.quantization_bit = None
        # LoRA variant validation + the quantization-compatibility guard.
        if self.lora_variant not in self.SUPPORTED_LORA_VARIANTS:
            raise ValueError(
                f"unsupported lora_variant {self.lora_variant!r}; "
                f"expected one of {self.SUPPORTED_LORA_VARIANTS}"
            )
        # Variants only ride the adapter methods (lora/qlora); full/freeze have no
        # adapter. Normalize a stray variant on a full-weight method to plain "lora"
        # rather than erroring (render.py ignores it anyway) so old/cloned configs
        # don't blow up. The UI also clears it on method switch.
        if self.finetuning_type in FULL_WEIGHT_METHODS:
            self.lora_variant = "lora"
        # HARD reject the invalid QLoRA × {DoRA, PiSSA} pairing BEFORE a billable
        # launch. quantization_bit is now settled above, so it's the precise signal
        # (a 4-bit base). DoRA/PiSSA need the full-precision weights a QLoRA base
        # lacks → PEFT fails at model-load. Caught a real race that burned 2 jobs.
        if self.quantization_bit is not None and self.lora_variant in self.QUANT_INCOMPATIBLE_VARIANTS:
            raise ValueError(
                f"LoRA variant {self.lora_variant!r} is incompatible with QLoRA "
                f"(a {self.quantization_bit}-bit base): {self.lora_variant} needs the "
                f"full-precision weight matrix, which a quantized base does not have. "
                f"Use plain LoRA for {self.lora_variant}, or pick rsLoRA / LoRA+ for QLoRA."
            )
        # Preference-loss (DPO family) validation. pref_loss only means anything for
        # stage=dpo; for any other stage normalize it to the default so a stray value
        # (old clone, API typo) can't leak a "dpo"-only key into a non-DPO config.
        if self.pref_loss not in self.SUPPORTED_PREF_LOSS:
            raise ValueError(
                f"unsupported pref_loss {self.pref_loss!r}; "
                f"expected one of {self.SUPPORTED_PREF_LOSS}"
            )
        if self.stage != "dpo":
            self.pref_loss = "sigmoid"
        # ORPO/SimPO are reference-free preference losses on the SAME preference
        # dataset as DPO — but they're LLaMA-Factory-only (the serverless managed
        # recipe exposes neither). Reject early so a serverless+orpo/simpo pick can't
        # reach a billable launch that would ignore the loss.
        if self.pref_loss in self.REFERENCE_FREE_PREF_LOSS and self.engine != "llama_factory":
            raise ValueError(
                f"pref_loss {self.pref_loss!r} (reference-free preference loss) is only "
                f"available on the llama_factory engine, not {self.engine!r}"
            )
        # SimPO target margin: clamp to a sane band. Only reaches the trainer for
        # pref_loss=simpo (render gates on it), but clamp unconditionally.
        self.simpo_gamma = min(max(float(self.simpo_gamma), _SIMPO_GAMMA_MIN), _SIMPO_GAMMA_MAX)
        # Efficiency knobs: clamp NEFTune; gate packing to SFT (it concatenates
        # samples, which is unsafe for preference/KTO/RL losses); both are
        # llama_factory-only (the serverless recipe has no such knobs).
        self.neftune_noise_alpha = min(
            max(float(self.neftune_noise_alpha), _NEFTUNE_ALPHA_MIN), _NEFTUNE_ALPHA_MAX
        )
        if self.packing and self.stage != "sft":
            raise ValueError(
                f"sequence packing is only supported for SFT (stage='sft'), not {self.stage!r} "
                "— packing concatenates samples, which corrupts preference/KTO/RL losses"
            )
        if (self.packing or self.enable_liger_kernel or self.neftune_noise_alpha > 0.0) \
                and self.engine != "llama_factory":
            raise ValueError(
                "efficiency knobs (NEFTune / Liger kernel / packing) are only available on the "
                f"llama_factory engine, not {self.engine!r}"
            )
        # Full-weight (full/freeze) gating + the LR safety net.
        if self.finetuning_type in FULL_WEIGHT_METHODS:
            # Phase-1 stage gate: full/freeze is SFT-only (DPO/KTO full-FT deferred).
            if self.stage not in self.FULL_FREEZE_STAGES:
                raise ValueError(
                    f"full-weight method {self.finetuning_type!r} is only supported for "
                    f"stage(s) {self.FULL_FREEZE_STAGES} in this phase, not {self.stage!r}"
                )
            # Clamp the learning rate: full-weight at the LoRA-scale 1e-4 diverges.
            # Snap anything above the ceiling down to the safe default (this is the
            # backend net for a UI/clone/API that forgot to lower the inherited LR).
            if not self.learning_rate or self.learning_rate > self.FULL_WEIGHT_MAX_LR:
                self.learning_rate = self.FULL_WEIGHT_DEFAULT_LR
            # freeze_trainable_layers only matters for freeze; keep it ≥1 so a
            # "freeze" run actually trains something.
            if self.finetuning_type == "freeze":
                self.freeze_trainable_layers = max(1, int(self.freeze_trainable_layers))
        # RLVR needs a verifiable reward: EITHER a PRESET (gsm8k/prime_math/
        # prime_code) OR a CUSTOM reward function (reward_function_id). Exactly one
        # must be set. For non-RLVR stages, clear both so they can't leak into a spec.
        if self.stage == "rlvr":
            has_preset = bool(self.preset_reward_function)
            has_custom = bool(self.reward_function_id)
            if has_preset and has_custom:
                raise ValueError(
                    "RLVR takes EITHER a preset reward function OR a custom "
                    "reward_function_id, not both"
                )
            if not has_preset and not has_custom:
                raise ValueError(
                    f"RLVR needs a preset reward function "
                    f"(one of {self.PRESET_REWARD_FUNCTIONS}) or a custom reward_function_id"
                )
            if has_preset and self.preset_reward_function not in self.PRESET_REWARD_FUNCTIONS:
                raise ValueError(
                    f"unknown preset reward function {self.preset_reward_function!r}; "
                    f"expected one of {self.PRESET_REWARD_FUNCTIONS}"
                )
            self.reward_model_id = ""  # RLVR has no AI-judge model
        elif self.stage == "rlaif":
            # RLAIF's reward is a NON-verifiable AI judge driven by a reward PROMPT.
            # There is NO preset (presets are verifiable-only) — the reward is ALWAYS
            # a reward_function_id pointing at a 'reward_prompt'-kind registry record.
            if self.preset_reward_function:
                raise ValueError(
                    "RLAIF uses an AI-judge reward prompt, not a preset reward function"
                )
            if not self.reward_function_id:
                raise ValueError(
                    "RLAIF needs a reward_function_id pointing at a registered reward prompt"
                )
            # reward_model_id (the judge model) is optional — the recipe has a
            # default judge; keep whatever was provided.
        else:
            self.preset_reward_function = ""
            self.reward_function_id = ""
            self.reward_model_id = ""

        # KTO loss weights: clamp to a sane range so a typo can't pass a degenerate
        # value (≤0 would zero/destabilize a class's loss). They only ever reach the
        # trainer for stage=kto (render.py gates on stage), but clamp unconditionally
        # so the stored value is always valid. Bounds mirror Hyperparams.bounds().
        self.kto_chosen_weight = min(max(float(self.kto_chosen_weight), _KTO_WEIGHT_MIN), _KTO_WEIGHT_MAX)
        self.kto_rejected_weight = min(max(float(self.kto_rejected_weight), _KTO_WEIGHT_MIN), _KTO_WEIGHT_MAX)
        # LoRA+ ratio: clamp like the KTO weights so a degenerate value can't reach
        # the trainer. Only emitted into the YAML for lora_variant=loraplus (render.py
        # gates on the variant), but clamp unconditionally so the stored value is valid.
        self.loraplus_lr_ratio = min(max(float(self.loraplus_lr_ratio), _LORAPLUS_RATIO_MIN), _LORAPLUS_RATIO_MAX)

    # Bounds used for server-side validation (kept here so UI + API agree).
    @staticmethod
    def bounds() -> dict[str, Any]:
        return {
            "loraRank": {"min": 1, "max": 256},
            "loraAlpha": {"min": 1, "max": 512},
            # LoRA+ B/A learning-rate ratio (only used when lora_variant=loraplus).
            # 1 = plain LoRA (no benefit); very high destabilizes the B matrix.
            "loraplusLrRatio": {"min": 1, "max": 128},
            "learningRate": {"min": 1e-6, "max": 1e-2},
            "numTrainEpochs": {"min": 0.1, "max": 50},
            "perDeviceTrainBatchSize": {"min": 1, "max": 64},
            "gradientAccumulationSteps": {"min": 1, "max": 128},
            "cutoffLen": {"min": 128, "max": 32768},
            "saveSteps": {"min": 1, "max": 100000},
            "ktoChosenWeight": {"min": _KTO_WEIGHT_MIN, "max": _KTO_WEIGHT_MAX},
            "ktoRejectedWeight": {"min": _KTO_WEIGHT_MIN, "max": _KTO_WEIGHT_MAX},
            # SimPO target reward margin γ (only used when pref_loss=simpo).
            "simpoGamma": {"min": _SIMPO_GAMMA_MIN, "max": _SIMPO_GAMMA_MAX},
            # NEFTune embedding-noise alpha (0 = off; ~5 is the paper sweet spot).
            "neftuneNoiseAlpha": {"min": _NEFTUNE_ALPHA_MIN, "max": _NEFTUNE_ALPHA_MAX},
        }


@dataclass
class DecodingParams:
    """Decoding settings for offline eval. The SAME values must apply to every
    candidate in a comparison (methodology). Defaults are deterministic (greedy).
    """

    backend: str = "vllm"  # "vllm" | "hf"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    seed: int = 42


# Eval token budget for REASONING models. Their <think>/CoT prefix routinely runs
# past the 256 default before reaching the answer; if it doesn't close in budget,
# eval.extract_answer strips the whole (unclosed) block and the answer scores 0
# (the RLVR spike hit exactly this). A reasoning model in a race raises the SHARED
# eval budget to this floor so the CoT closes — fairness is preserved (every racer
# still decodes with identical settings) and it only ever RAISES a too-low value.
REASONING_EVAL_MAX_NEW_TOKENS = 512


def reasoning_eval_floor(model_ids: list[str], requested: int) -> int:
    """The eval max_new_tokens to actually use: `requested`, but raised to the
    reasoning floor if ANY model in the comparison is a reasoning family (so its
    <think> block closes). Never lowers an explicit higher request. Pure +
    unit-testable; the caller logs when it bumps."""
    if requested >= REASONING_EVAL_MAX_NEW_TOKENS:
        return requested
    if any((m := get_model(mid)) is not None and m.reasoning for mid in model_ids):
        return REASONING_EVAL_MAX_NEW_TOKENS
    return requested
