# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Goal + dataset profile + effort → an EXACT, launchable race plan.

This is the one genuinely-new ML bit behind the Guided Fine-tuning agent: turning
a plain-language goal (captured upstream) and the deterministic dataset profile
into a list of RaceModel entries the user can approve and launch. It is the
"run-config advisor" the platform has long flagged, made concrete.

DISCIPLINE (the lesson from shipping engine-rejected configs, and the AutoML-HPO
literature — AgentHPO arXiv:2402.01881, Zhang arXiv:2312.04528):
  * 100% RULE-BASED for every load-bearing decision — objective, engine, method,
    which models, the size ladder, hyperparameters, and every gate. No LLM picks a
    model or a number here. (pitcrew.py calls a NARRATION model afterwards to write
    the plain-language "why" over this already-final plan; that text is never parsed
    back into config.)
  * The OBJECTIVE is data-shape-FIXED, taken verbatim from recommend_objective() —
    a preference dataset can only train DPO, etc. The agent never offers a mismatch.
  * Spend priority is MODEL-SIZE LADDER → FAMILY diversity → one METHOD/VARIANT arm
    → hyperparameter sweep. The scaling-law work (Rectified Scaling arXiv:2402.02314,
    joint scaling arXiv:2402.17193) says race a size ladder rather than bet on the
    biggest, and the LoRA-LR work (Lee et al. arXiv:2602.04998 "Learning Rate
    Matters: Vanilla LoRA May Suffice", corroborated by He et al. arXiv:2601.22708
    "A Unified Study of LoRA Variants") shows the variant axis buys little (~1-2%)
    once the LR is tuned PER METHOD — and because the optimal LR is method-specific,
    a variant arm run at LoRA's recipe LR would likely UNDER-perform, not merely tie.
    So models/sizes + family diversity come FIRST; DoRA/full are optional A/B arms the
    leaderboard decides, never silent default switches.
  * Every cell is validated against the catalog gates (engine×stage×method, the ≤2B
    full/freeze size gate, gated-model HF-token need, reasoning-family fit) BEFORE it
    enters the plan; an ineligible candidate is skipped, never launched.

V1 SCOPE: SFT / DPO / KTO only. RLVR + RLAIF need a reward function/prompt collected
upstream (and RLAIF hard-raises without one), so a dataset whose shape implies them
returns an UNSUPPORTED plan with a clear message rather than an un-launchable race.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import (
    FULL_FREEZE_MAX_PARAMS_B,
    FULL_WEIGHT_METHODS,
    Hyperparams,
    ModelSpec,
    _instance_for,
    get_model,
)
from .limits import max_models_per_race
from .profiler import recommend_eval_strategy, recommend_objective
from .race import RaceModel
from .recommend import suggest_config

# Effort is an "UP TO" CEILING on the number of jobs, NOT a fixed count. The planner
# fills arms in priority order and STOPS EARLY when the next arm would add no new
# signal, so a small/simple dataset on "thorough" yields fewer than the ceiling (and
# says why) rather than padding the race with redundant, billable arms. The ceiling is
# clamped to the per-tenant model cap so a plan can never exceed what start_race accepts.
#   quick    → up to 4    balanced → up to 8    thorough → up to 16
# (The legacy keys easy/medium/huge remain accepted aliases so old sessions/tests and
# any cached client payloads keep working; they map to the same ceilings.)
EFFORT_JOBS = {"quick": 4, "balanced": 8, "thorough": 16,
               "easy": 4, "medium": 8, "huge": 16}

# v1 supports only these objectives in the guided flow (SFT/DPO/KTO). rlvr/rlaif
# are recognised but routed to an UNSUPPORTED plan (they need reward setup upstream).
SUPPORTED_OBJECTIVES = ("sft", "dpo", "kto")

# --- Reasoning-base selection (arXiv:2509.22193 "Scale or Reason?") --------------
# CHOOSING a reasoning (CoT/<think>) base is gated on OUTPUT FORMAT + SIZE, not on the
# data's scaffold rate. The controlled 0.5B-14B study shows a reasoning base pays off
# on OPEN-ENDED long-form answers and reaches the accuracy/cost frontier mainly at
# ≥7B; on short/closed answers (label/json/short-numeric) small reasoning models
# "overthink" (wrong traces 1.5-2x longer) — so we gate them OUT there regardless of
# scaffold. A reasoning base is eligible only for SFT on a text task whose answers are
# long enough to be genuinely open-ended.
_REASONING_MIN_ANSWER_P95_WORDS = 60   # goldWordLen.p95 floor for "open-ended" text
# NOTE (decoupled — see _is_reasoning_dataset vs _reasoning_base_fits): the DATA having
# a <think>/reasoning scaffold is a SEPARATE signal used only to raise the eval token
# floor so an unclosed <think> can't zero an otherwise-correct answer. It must NOT, by
# itself, pull reasoning FAMILIES into the model set — that is the format+size gate's job.
_REASONING_SCAFFOLD_THRESHOLD = 0.2

# --- FULL fine-tuning arm: DATA-gated, not tier-gated (arXiv:2402.17193, 2405.09673) --
# Full-weight FT's quality edge over LoRA appears mainly as dataset size + domain shift
# rise; on small/narrow data it forgets more and collapses generation diversity while a
# well-configured LoRA matches it. So offer a FULL A/B arm only when the data is large
# AND a large-domain-shift (structured) target, and SUPPRESS it entirely on small sets.
# Thresholds are conservative engineering choices (the paper's true crossover is
# task-specific — "no universal answer"), exposed as named knobs to tune from outcomes.
_FULL_ARM_MIN_ROWS = 50000     # offer the FULL arm only at/above this row count
_FULL_ARM_SUPPRESS_ROWS = 5000  # below this, never offer full/freeze at all
# Task shapes that represent a large enough domain shift to justify full FT (structured
# / low-overlap-with-pretraining targets). Uses the profiler's dominantTask vocabulary
# {json, numeric, label, text} — note "code" is NOT a dominantTask value (code answers
# classify as "text"), so code intent is read from the scaffold code-fence rate instead.
_FULL_ARM_TASKS = ("json", "numeric")
_FULL_ARM_CODE_FENCE_RATE = 0.5  # or: mostly code-fenced text answers ⇒ structured target
# Only trust dominantTask for these gates when the dataset is task-consistent enough
# (a 55/45 mix shouldn't trigger a task-shape gate).
_TASK_CONSISTENCY_FLOOR = 0.8


@dataclass
class PlannedModel:
    """One model in the plan: the launchable RaceModel plus display metadata + the
    deterministic plain-language ROLE the narration layer expands on."""
    race_model: RaceModel
    display_name: str
    params_b: float
    method: str
    variant: str
    role: str  # plain-language why-this-one (e.g. "a smaller, cheaper option")

    @property
    def entry_key(self) -> str:
        """STABLE per-arm id: the SAME model can appear as several distinct arms
        (plain LoRA + DoRA + full fine-tuning; or DPO + ORPO), so `modelId` alone is
        NOT unique. This mirrors race.entry_key_for's axes so the UI can key rows and
        target remove/edit at the RIGHT arm (not collapse/clobber all arms of a model)."""
        hp = self.race_model.hp
        parts = [self.race_model.model_id, self.method, self.variant, hp.pref_loss]
        return "::".join(parts)

    def display_label(self) -> str:
        """A human label that DISTINGUISHES arms of the same model, so three
        Qwen2.5-1.5B arms don't all read identically ('Qwen2.5 1.5B',
        'Qwen2.5 1.5B · DoRA', 'Qwen2.5 1.5B · full fine-tuning')."""
        extra = []
        if self.method == "full":
            extra.append("full fine-tuning")
        elif self.method == "freeze":
            extra.append("partial fine-tuning")
        elif self.method == "qlora":
            extra.append("QLoRA")
        if self.variant and self.variant != "lora":
            extra.append(self.variant.upper() if self.variant in ("dora", "pissa") else self.variant)
        pl = self.race_model.hp.pref_loss
        if pl and pl != "sigmoid":
            extra.append(pl.upper())
        return self.display_name + (f" · {' · '.join(extra)}" if extra else "")

    def to_view(self) -> dict[str, Any]:
        hp = self.race_model.hp
        return {
            "entryKey": self.entry_key,   # stable per-arm id (UI trackBy + edit target)
            "modelId": self.race_model.model_id,
            "displayName": self.display_name,
            "label": self.display_label(),  # distinguishes arms of the same model
            "paramsB": self.params_b,
            "instanceType": self.race_model.instance_type,
            "method": self.method,
            "variant": self.variant,
            "stage": hp.stage,
            "learningRate": hp.learning_rate,
            "loraRank": hp.lora_rank if self.method in ("lora", "qlora") else None,
            "role": self.role,
        }


@dataclass
class RacePlan:
    """A complete, launchable (or explicitly-unsupported) race plan."""
    supported: bool
    objective: str                       # sft | dpo | kto | rlvr | rlaif
    engine: str
    effort: str
    job_budget: int                      # the effort CEILING ("up to N"), after cap-clamp
    rank_metric: str
    detected_task: str
    planned: list[PlannedModel] = field(default_factory=list)
    gates_applied: list[str] = field(default_factory=list)
    reason: str = ""                     # set when supported is False
    # The ceiling was an "up to N" cap, not a quota: the planner fills arms only while
    # each adds new signal, so `capped` is True when it stopped BELOW the ceiling (no
    # more distinct, useful arms exist for this dataset). The UI surfaces this as "we
    # filled M of N — more wouldn't add anything" instead of padding with redundant arms.
    capped: bool = False                 # True ⇒ fewer arms than the ceiling, deliberately

    @property
    def race_models(self) -> list[RaceModel]:
        """The launchable RaceModel list (hand straight to race.start_race)."""
        return [p.race_model for p in self.planned]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "objective": self.objective,
            "engine": self.engine,
            "effort": self.effort,
            "jobBudget": self.job_budget,
            "rankMetric": self.rank_metric,
            "detectedTask": self.detected_task,
            "models": [p.to_view() for p in self.planned],
            "gatesApplied": self.gates_applied,
            "reason": self.reason,
            # How many arms we actually filled vs the ceiling the user asked for, and
            # whether we deliberately stopped short (no more useful arms). The UI uses
            # this to show "raced M of up-to-N; more wouldn't add signal".
            "meaningfulCount": len(self.planned),
            "ceiling": self.job_budget,
            "capped": self.capped,
        }


# --- Preferred model ladders -------------------------------------------------
# Ordered families, each a size ladder (smallest→largest catalog rung). The planner
# adds a family's small rung first, then its next rung, growing breadth (families) and
# depth (sizes) as the job ceiling allows — so higher ceilings fill with GENUINE
# diversity, and when the catalog runs out the plan honestly returns fewer than the
# ceiling rather than padding. Each id is filtered to whatever is actually eligible
# (exists, not gated-without-token, reasoning-fit). NON-reasoning families deliberately
# exclude every Qwen3 model: ModelSpec.reasoning matches the "qwen3" substring for ALL
# qwen3 templates (including qwen3_nothink), so qwen3-* is treated as a reasoning family
# — the single source of truth shared with eval's reasoning token floor.
_NONREASONING_LADDER = [
    ("Qwen2.5", ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b"]),
    ("Granite", ["granite-3.1-2b", "granite-3.1-8b"]),
    ("Phi", ["phi-3.5-mini", "phi-4"]),
    ("MiniCPM", ["minicpm4-0.5b", "minicpm4-8b"]),
    ("Llama", ["llama-3.2-3b", "llama-3.1-8b"]),   # gated — only if hf_token_ok
    ("InternLM", ["internlm2.5-7b"]),
    ("GLM", ["glm-4-9b"]),
    ("Mistral", ["mistral-7b"]),                    # gated
    ("Gemma", ["gemma-2-9b"]),                      # gated
]
# Reasoning ladders (used only when a reasoning BASE fits — see _reasoning_base_fits).
_REASONING_LADDER = [
    ("Qwen3", ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"]),
    ("DeepSeek-R1", ["deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-7b",
                     "deepseek-r1-distill-qwen-14b"]),
    ("Qwen2.5", ["qwen2.5-1.5b", "qwen2.5-7b"]),   # non-reasoning sibling for A/B contrast
]
# A tiny (≤2B) model to carry the optional full-weight SFT arm.
_FULL_WEIGHT_TINY = "qwen2.5-1.5b"


def _train_rows(prof: dict[str, Any], objective: str) -> int:
    """Usable training-row count for the dataset's shape (the recommender input)."""
    if objective == "dpo":
        return int((prof.get("preference") or {}).get("pairs", 0) or 0)
    if objective == "kto":
        return int((prof.get("kto") or {}).get("rows", 0) or 0)
    if objective in ("rlvr", "rlaif"):
        return int((prof.get(objective) or {}).get("rows", 0) or 0)
    # sft: use the TRUE (uncapped) row count, not the ≤5000 profiler sample, so the
    # rank/LR heuristics scale to the real dataset size (a 50k-row set should get
    # rank 32, not the rank 16 a 5000-sample would imply).
    train = prof.get("train") or {}
    return int(train.get("rows") or train.get("parsed", 0) or 0)


def needs_raised_eval_floor(prof: dict[str, Any], objective: str) -> bool:
    """The DATA's gold answers carry a <think>/reasoning scaffold often enough (or it's
    a math-RLVR set) that a model fine-tuned on it will emit a reasoning block at eval
    time — so the eval token floor must be raised for that block to CLOSE, or an unclosed
    <think> zeros an otherwise-correct answer (the documented bug). This is a property of
    the DATA (what the tuned model learns to emit), so it holds for ANY base family, NOT
    just reasoning-family bases. Deliberately DECOUPLED from reasoning-base SELECTION
    (that's _reasoning_base_fits): scaffolded-but-short-answer data should still raise the
    eval floor even though we pick an instruct base for it."""
    if objective == "rlvr":
        if ((prof.get("rlvr") or {}).get("numericGroundTruthRate") or 0) >= 0.8:
            return True
    for key in ("train", "eval"):
        sc = ((prof.get(key) or {}).get("scaffold") or {})
        if (sc.get("rate") or 0) > _REASONING_SCAFFOLD_THRESHOLD:
            return True
        if "think" in (sc.get("patterns") or {}):
            return True
    return False


def _reasoning_base_fits(prof: dict[str, Any], objective: str) -> bool:
    """Whether a REASONING (CoT/<think>) base family should be pulled into the model set
    — a FORMAT + SIZE gate, NOT the data's scaffold rate (arXiv:2509.22193 "Scale or
    Reason?"). A reasoning base pays off only on OPEN-ENDED long-form answers and reaches
    the accuracy/cost frontier mainly at ≥7B; on short/closed answers it "overthinks"
    (wrong traces 1.5-2x longer) and costs accuracy + tokens. So a reasoning base is
    eligible only for SFT on a task-consistent TEXT dataset whose answers are long enough
    to be genuinely open-ended. Math-RLVR keeps its own numeric-ground-truth path.

    Why format, not scaffold: scaffold.rate says the DATA contains <think> traces, which
    is a signal about eval tokens (see needs_raised_eval_floor), NOT about whether a
    reasoning BASE helps the TASK — those are different questions, so they use different
    signals."""
    if objective == "rlvr":
        return ((prof.get("rlvr") or {}).get("numericGroundTruthRate") or 0) >= 0.8
    # Reasoning-base evidence is SFT-only; DPO/KTO with a reasoning base is untested, so
    # stay on instruct bases for the preference objectives.
    if objective != "sft":
        return False
    f = prof.get("eval") or prof.get("train") or {}
    task = f.get("dominantTask", "text")
    consistency = f.get("taskConsistency", 1.0)
    if task != "text" or consistency < _TASK_CONSISTENCY_FLOOR:
        return False  # short/closed or mixed → instruct base
    p95 = ((f.get("goldWordLen") or {}).get("p95")) or 0
    return p95 >= _REASONING_MIN_ANSWER_P95_WORDS


def _full_arm_fits(prof: dict[str, Any], rows: int, objective: str) -> bool:
    """Whether a FULL fine-tuning A/B arm is justified for this dataset (arXiv:2402.17193,
    2405.09673). Full-weight FT's edge over LoRA appears only when the dataset is LARGE
    and a large-domain-shift (structured) target; on small/narrow data it forgets more
    while LoRA matches it. So require: SFT, rows ≥ _FULL_ARM_MIN_ROWS, and a structured
    target — dominantTask ∈ {json, numeric} (task-consistent) OR mostly code-fenced text
    answers (code answers classify as 'text', so code intent is read from the scaffold
    code-fence rate, NOT a 'code' dominantTask — which the profiler never emits)."""
    if objective != "sft" or not rows or rows < _FULL_ARM_MIN_ROWS:
        return False
    f = prof.get("train") or prof.get("eval") or {}
    task = f.get("dominantTask", "text")
    consistency = f.get("taskConsistency", 1.0)
    if task in _FULL_ARM_TASKS and consistency >= _TASK_CONSISTENCY_FLOOR:
        return True
    # Code-heavy text answers (fenced code) are a structured/large-domain-shift target too.
    code_fence_rate = ((f.get("scaffold") or {}).get("patterns") or {}).get("code_fence", 0) or 0
    return code_fence_rate >= _FULL_ARM_CODE_FENCE_RATE


def _eligible(model_id: str, *, hf_token_ok: bool, reasoning_ok: bool) -> ModelSpec | None:
    """Return the ModelSpec if this candidate may be raced under the current gates,
    else None. Gates: model must exist; gated models need an HF token; reasoning
    families are excluded unless the data justifies them."""
    spec = get_model(model_id)
    if spec is None:
        return None
    if spec.gated and not hf_token_ok:
        return None
    if spec.reasoning and not reasoning_ok:
        return None
    return spec


def _build_hp(spec: ModelSpec, rows: int, has_val: bool, objective: str,
              method: str, variant: str) -> Hyperparams:
    """Deterministic, bounds-clamped hyperparameters for one (model, method),
    via the trusted recommender, with the objective/variant set on top."""
    # Pass the objective so the recommender picks an objective-appropriate LoRA LR
    # (SFT ~2e-4 vs DPO/KTO ~5e-6) — a preference run must NOT inherit the SFT LR.
    rec = suggest_config(spec, rows, has_val, finetuning_type=method, objective=objective)
    hp = rec.hp
    hp.stage = objective
    hp.engine = "llama_factory"  # v1: every supported objective runs on LLaMA-Factory
    if method in ("lora", "qlora"):
        hp.lora_variant = variant
    return hp


def _instance(spec: ModelSpec, method: str) -> str:
    """The instance start_race would auto-pick (mirrored so the cost estimate and the
    launch agree): full/freeze re-resolve method-aware (heavier GPU); else the
    catalog's fp16-LoRA suggested instance."""
    if method in FULL_WEIGHT_METHODS:
        return _instance_for(spec.params_b, method)
    return spec.suggested_instance


def _make(spec: ModelSpec, rows: int, has_val: bool, objective: str, *,
          method: str = "lora", variant: str = "lora", pref_loss: str = "sigmoid",
          role: str) -> PlannedModel:
    hp = _build_hp(spec, rows, has_val, objective, method, variant)
    if objective == "dpo" and pref_loss != "sigmoid":
        hp.pref_loss = pref_loss
    rm = RaceModel(model_id=spec.id, hp=hp, instance_type=_instance(spec, method))
    return PlannedModel(race_model=rm, display_name=spec.display_name,
                        params_b=spec.params_b, method=method, variant=variant, role=role)


def plan_race(prof: dict[str, Any], effort: str, *, hf_token_ok: bool = False,
              max_jobs: int | None = None) -> RacePlan:
    """Build a launchable race plan from a dataset profile + an effort ceiling.

    `prof` is profiler.profile_dataset(split_id). `effort` ∈ {quick, balanced, thorough}
    (legacy easy/medium/huge accepted) — an "up to N" CEILING, not a fixed count.
    `hf_token_ok` says whether a working HF token is present (gates gated models).
    `max_jobs` (optional) is an EXACT ceiling override (the manual auto-fill picker lets
    the user pick any count, e.g. 6) — it takes precedence over the effort tier so the
    plan fills up to exactly that many. Both are clamped to max_models_per_race().

    Deterministic + total: never raises. An unsupported objective (rlvr/rlaif in v1)
    or an empty eligible pool yields a plan with supported=False / no models and a
    plain-language reason, so the caller surfaces a clear message instead of a crash.
    """
    objective = (recommend_objective(prof) or {}).get("objective", "sft")
    es = recommend_eval_strategy(prof) or {}
    rank_metric = es.get("rankMetric", "token_f1")
    detected_task = es.get("detectedTask", "text")
    engine = "llama_factory"

    effort = effort if effort in EFFORT_JOBS else "balanced"
    ceiling = int(max_jobs) if max_jobs and max_jobs > 0 else EFFORT_JOBS[effort]
    budget = min(ceiling, max_models_per_race())

    plan = RacePlan(supported=True, objective=objective, engine=engine, effort=effort,
                    job_budget=budget, rank_metric=rank_metric, detected_task=detected_task)

    # v1 RL exclusion — recognised, but routed out with a clear reason.
    if objective not in SUPPORTED_OBJECTIVES:
        plan.supported = False
        plan.reason = (
            "This dataset looks like a reinforcement-learning task, which needs a "
            "reward to be set up first. Guided Fine-tuning doesn't handle that yet — "
            "use the full Fine-tune page for reward-based training."
        )
        plan.gates_applied.append(f"objective '{objective}' not supported in guided v1")
        return plan

    rows = _train_rows(prof, objective)
    has_val = bool(prof.get("hasVal"))

    # Reasoning-BASE selection (format+size) is DECOUPLED from the eval-token-floor
    # (scaffold presence). needs_raised_eval_floor drives the eval budget at launch for
    # ANY base; _reasoning_base_fits decides whether reasoning FAMILIES enter the set.
    reasoning_ok = _reasoning_base_fits(prof, objective)
    if reasoning_ok:
        plan.gates_applied.append(
            "reasoning families allowed (open-ended long-form text task — format+size gate)")
    if needs_raised_eval_floor(prof, objective):
        plan.gates_applied.append("eval token budget will be raised (data shows chain-of-thought)")

    ladder = _REASONING_LADDER if reasoning_ok else _NONREASONING_LADDER

    # --- Walk a deterministic priority list of "arms", filling UP TO the ceiling. ---
    # Each arm yields at most one PlannedModel; ineligible/duplicate arms are skipped.
    # The list is finite and ordered by expected value, so the planner naturally STOPS
    # EARLY (returns fewer than the ceiling) when no more distinct, useful arm exists —
    # never padding a race with redundant, billable jobs.
    chosen: list[PlannedModel] = []
    seen: set[str] = set()  # entry keys to de-dup — mirrors race.entry_key_for's axes

    def _key(model_id: str, method: str, variant: str, pref_loss: str) -> str:
        # Include pref_loss so DPO-sigmoid and DPO-orpo of the SAME model are DISTINCT
        # entries (race.entry_key_for keys on it too); otherwise the ORPO arm would
        # collide with the plain-DPO rung of that model and be silently dropped.
        return f"{model_id}::{method}::{variant}::{pref_loss}"

    def _add(spec: ModelSpec, *, method="lora", variant="lora", pref_loss="sigmoid", role: str) -> bool:
        if len(chosen) >= budget:
            return False
        k = _key(spec.id, method, variant, pref_loss)
        if k in seen:
            return False
        seen.add(k)
        chosen.append(_make(spec, rows, has_val, objective, method=method,
                            variant=variant, pref_loss=pref_loss, role=role))
        return True

    small_role = "a smaller, faster, cheaper model — often good enough"
    mid_role = "a larger model with more capacity (costs more — we race it to see if it's worth it)"
    div_small = "a different model family, for variety"
    div_mid = "a larger model from a different family"
    eligible_ladder = [
        (family, [s for mid in rungs
                  if (s := _eligible(mid, hf_token_ok=hf_token_ok, reasoning_ok=reasoning_ok))])
        for family, rungs in ladder
    ]
    max_depth = max((len(rungs) for _f, rungs in eligible_ladder), default=0)

    def _fill_ladder(limit: int) -> None:
        """Add ladder rungs up to `limit` total arms: round 0 = each family's smallest
        rung (breadth), later rounds = the next size up (depth). The primary family's two
        smallest rungs carry the 'smaller vs larger' framing; the rest are diversity."""
        for depth in range(max_depth):
            if len(chosen) >= limit:
                return
            for fam_i, (_family, rungs) in enumerate(eligible_ladder):
                if len(chosen) >= limit:
                    return
                if depth >= len(rungs):
                    continue
                if fam_i == 0:
                    role = small_role if depth == 0 else mid_role
                else:
                    role = div_small if depth == 0 else div_mid
                _add(rungs[depth], role=role)

    # RESERVE slots for the technique A/B arm(s) so the (now 9-family) ladder can't crowd
    # them out — the research wants the technique A/B to enter by the medium tier, not
    # only at the top. Count the arms that will apply, then keep at least 2 ladder rungs
    # (a technique A/B is only meaningful next to its plain-LoRA baseline sibling).
    n_technique = 0
    if objective == "sft":
        n_technique = 1  # DoRA A/B
        if _full_arm_fits(prof, rows, objective):
            n_technique += 1
    elif objective == "dpo":
        n_technique = 1  # ORPO A/B
    reserve = min(n_technique, max(0, budget - 2))

    # 1) MODEL-SIZE LADDER + FAMILY diversity — highest-value spend, filled FIRST (but
    #    leaving `reserve` slots for the technique arms below).
    _fill_ladder(budget - reserve)

    # 2) TECHNIQUE / VARIANT A/B arms — leaderboard A/B experiments the research supports,
    #    NEVER default switches (the variant axis buys ~1-2% once LR is sane — 2602.04998):
    #      - SFT → a DoRA arm on the primary small model (rides plain LoRA; ~+0.5-4% at
    #        r16-32 — 2402.09353), then a data-gated FULL arm on a ≤2B model when the
    #        dataset is large AND a structured/large-domain-shift target.
    #      - DPO → a reference-free ORPO arm (cheaper; a genuinely different algorithm).
    if chosen and objective == "sft":
        primary_spec = get_model(chosen[0].race_model.model_id)
        if primary_spec and chosen[0].method == "lora" and chosen[0].variant == "lora":
            _add(primary_spec, method="lora", variant="dora",
                 role="the same small model with an advanced adapter (DoRA) — "
                      "can recover more quality at the same cost")
        if _full_arm_fits(prof, rows, objective):
            ft = _eligible(_FULL_WEIGHT_TINY, hf_token_ok=hf_token_ok, reasoning_ok=reasoning_ok)
            if ft and "full" in (ft.allowed_methods or ()):
                if _add(ft, method="full", variant="lora",
                        role=(f"full fine-tuning of a small model — with {rows:,} rows on a "
                              "structured task it can beat LoRA (costs more; we race it to find out)")):
                    plan.gates_applied.append(
                        f"full-FT arm offered (large structured dataset: {rows:,} rows)")
    elif chosen and objective == "dpo":
        primary_spec = get_model(chosen[0].race_model.model_id)
        if primary_spec:
            _add(primary_spec, method="lora", variant="lora", pref_loss="orpo",
                 role="the same model trained with ORPO — a cheaper, reference-free "
                      "preference method we race against the standard one")
    # KTO gets the size ladder only — no variant/technique arm has literature support on
    # the KTO objective (all variant evidence is SFT-style), so we stay conservative.

    # 3) BACKFILL — if the technique arms didn't materialize (e.g. DoRA collided, or the
    #    full arm gated out at the last moment), fill any reserved-but-unused slots from
    #    the ladder so we still honor the ceiling with real, distinct arms.
    if len(chosen) < budget:
        _fill_ladder(budget)

    plan.planned = chosen
    # capped = we deliberately stopped BELOW the ceiling (no more distinct useful arms).
    plan.capped = bool(chosen) and len(chosen) < budget
    if objective == "sft" and rows and rows < _FULL_ARM_SUPPRESS_ROWS:
        plan.gates_applied.append(
            f"full/freeze suppressed (small dataset: {rows:,} rows — LoRA matches it without forgetting)")
    if not chosen:
        plan.supported = False
        plan.reason = (
            "I couldn't assemble a set of models to race for this dataset. "
            "If your task needs specific (gated) models, add a Hugging Face token in "
            "Settings, or use the full Fine-tune page."
        )
    return plan


def eligible_models(prof: dict[str, Any], *, hf_token_ok: bool = False) -> list[dict[str, Any]]:
    """The catalog models a user MAY add to this dataset's race (the swap/add pool).

    Same gates the planner applies: the objective must be supported, gated models
    need a token, and reasoning families are offered only when the data justifies
    them. Returned smallest-first (a sensible default order for the picker), each
    with the plain display facts the UI needs. Empty for an unsupported objective."""
    objective = (recommend_objective(prof) or {}).get("objective", "sft")
    if objective not in SUPPORTED_OBJECTIVES:
        return []
    reasoning_ok = _reasoning_base_fits(prof, objective)
    from .catalog import list_models

    out: list[dict[str, Any]] = []
    for m in list_models():
        spec = get_model(m["id"])
        if spec is None:
            continue
        if spec.gated and not hf_token_ok:
            continue
        if spec.reasoning and not reasoning_ok:
            continue
        out.append({
            "modelId": spec.id,
            "displayName": spec.display_name,
            "paramsB": spec.params_b,
            "family": spec.family,
        })
    return sorted(out, key=lambda r: r["paramsB"])


def specs_from_plan(plan: RacePlan) -> list[dict[str, Any]]:
    """The editable "specs" for a plan — one per planned model, carrying only the
    distinguishing axes (modelId, method, variant, prefLoss) + the display role. The
    session persists these so the review screen can edit the SET (remove an entry,
    add a model) and the launch rebuilds hyperparameters deterministically from them
    — the backend never trusts a client-sent hyperparameter."""
    out: list[dict[str, Any]] = []
    for p in plan.planned:
        out.append({
            "modelId": p.race_model.model_id,
            "method": p.method,
            "variant": p.variant,
            "prefLoss": p.race_model.hp.pref_loss,
            "role": p.role,
        })
    return out


def build_models_from_specs(prof: dict[str, Any], specs: list[dict[str, Any]], *,
                            hf_token_ok: bool = False) -> tuple[list[PlannedModel], list[str]]:
    """Rebuild gate-valid PlannedModels from a (possibly user-edited) spec list.

    Returns (planned, skipped_ids). Each spec is run through the SAME recipe — the
    objective comes from the DATA (never the spec), hyperparameters come from
    suggest_config, and every (engine,stage,method,variant) cell is re-validated —
    so a hand-edited plan can never become an invalid billable launch. A spec that
    fails a gate (unknown model, gated-without-token, reasoning-misfit, full/freeze
    on a >2B model, over the cap) is dropped into `skipped`. De-duped by the same
    axes race.entry_key_for uses; clamped to max_models_per_race()."""
    objective = (recommend_objective(prof) or {}).get("objective", "sft")
    if objective not in SUPPORTED_OBJECTIVES:
        return [], [s.get("modelId", "?") for s in specs]
    rows = _train_rows(prof, objective)
    has_val = bool(prof.get("hasVal"))
    reasoning_ok = _reasoning_base_fits(prof, objective)
    cap = max_models_per_race()

    built: list[PlannedModel] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for s in specs:
        mid = s.get("modelId", "")
        method = s.get("method", "lora")
        variant = s.get("variant", "lora")
        pref_loss = s.get("prefLoss", "sigmoid")
        key = f"{mid}::{method}::{variant}::{pref_loss}"
        if key in seen:
            continue
        seen.add(key)
        spec = _eligible(mid, hf_token_ok=hf_token_ok, reasoning_ok=reasoning_ok)
        if spec is None or len(built) >= cap:
            skipped.append(mid)
            continue
        # Re-validate the cell's gates (full/freeze size + SFT-only, DoRA≠qlora, etc.)
        # by constructing it; a bad combo is skipped, never launched.
        if method in FULL_WEIGHT_METHODS:
            if (spec.params_b > FULL_FREEZE_MAX_PARAMS_B or method not in (spec.allowed_methods or ())
                    or objective != "sft"):
                skipped.append(mid)
                continue
        try:
            pm = _make(spec, rows, has_val, objective, method=method, variant=variant,
                      pref_loss=pref_loss, role=s.get("role", "you chose this model"))
        except (ValueError, TypeError):
            skipped.append(mid)
            continue
        built.append(pm)
    return built, skipped
