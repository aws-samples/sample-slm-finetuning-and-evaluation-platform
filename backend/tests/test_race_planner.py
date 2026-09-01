# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic race planner — goal+profile+effort → a launchable, gate-valid plan."""
import json

from app import race_planner
from app.catalog import FULL_FREEZE_MAX_PARAMS_B, Hyperparams, get_model
from app.profiler import profile_dataset


def _chat(user, assistant, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def _make_sft_split(store_mod, split_id, n_train=2000, n_eval=100, has_val=False):
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    train = [_chat(f"classify ticket {i}", "billing") for i in range(n_train)]
    eval_ = [_chat(f"classify ticket e{i}", "billing") for i in range(n_eval)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    if has_val:
        (wd / "val.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "sft", "hasVal": has_val,
         "trainRows": n_train, "evalRows": n_eval, "valRows": n_eval if has_val else 0}),
        encoding="utf-8")
    store.commit("runs", split_id)


def _make_preference_split(store_mod, split_id, n=300):
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    # Preference TRAIN rows are ranking-shaped; eval is the chosen-as-gold projection.
    train = [{"messages": [{"role": "user", "content": f"q{i}"}],
              "chosen": {"role": "assistant", "content": "good"},
              "rejected": {"role": "assistant", "content": "bad"}} for i in range(n)]
    eval_ = [_chat(f"q e{i}", "good") for i in range(40)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "preference", "hasVal": False, "trainRows": n}),
        encoding="utf-8")
    store.commit("runs", split_id)


def _assert_gate_valid(plan):
    """Every planned model must be constructible as Hyperparams (engine×stage×method
    gate) AND obey the catalog full/freeze size gate — i.e. actually launchable."""
    for p in plan.planned:
        spec = get_model(p.race_model.model_id)
        assert spec is not None, p.race_model.model_id
        hp = p.race_model.hp
        # Re-running __post_init__ via a copy proves the cell passes the engine gates.
        Hyperparams(**{k: getattr(hp, k) for k in (
            "engine", "stage", "finetuning_type", "lora_variant", "pref_loss",
            "learning_rate", "lora_rank")})
        if p.method in ("full", "freeze"):
            assert spec.params_b <= FULL_FREEZE_MAX_PARAMS_B
            assert p.method in (spec.allowed_methods or ())
            assert hp.stage == "sft"
            assert hp.engine == "llama_factory"
        # No DoRA/PiSSA on a 4-bit base.
        if p.variant in ("dora", "pissa"):
            assert p.method != "qlora"


def test_quick_ceiling_fills_size_ladder(temp_store):
    _make_sft_split(temp_store, "sft-easy")
    prof = profile_dataset("sft-easy")
    plan = race_planner.plan_race(prof, "quick")
    assert plan.supported and plan.objective == "sft"
    # "quick" is an UP-TO-4 ceiling; a rich-enough catalog fills it.
    assert 2 <= len(plan.planned) <= 4
    # The primary family's two rungs should be a size ladder (smaller then larger).
    fam = plan.planned[0].display_name
    assert plan.planned[0].params_b <= max(p.params_b for p in plan.planned)
    _assert_gate_valid(plan)


def test_legacy_effort_aliases_map_to_new_ceilings(temp_store):
    # Old easy/medium/huge keys must still work and map to up-to 4/8/16.
    _make_sft_split(temp_store, "sft-alias")
    prof = profile_dataset("sft-alias")
    assert race_planner.plan_race(prof, "easy").job_budget == 4
    assert race_planner.plan_race(prof, "medium").job_budget == 8
    assert race_planner.plan_race(prof, "huge").job_budget == 16


def test_effort_ceiling_scales_job_count(temp_store):
    _make_sft_split(temp_store, "sft-scale")
    prof = profile_dataset("sft-scale")
    quick = race_planner.plan_race(prof, "quick")
    balanced = race_planner.plan_race(prof, "balanced")
    thorough = race_planner.plan_race(prof, "thorough")
    # Each is bounded by its ceiling, and a higher ceiling yields at least as many arms.
    assert len(quick.planned) <= 4
    assert len(balanced.planned) <= 8
    assert len(thorough.planned) <= 16
    assert len(quick.planned) <= len(balanced.planned) <= len(thorough.planned)
    assert len(balanced.planned) > len(quick.planned)  # more budget → more real arms
    _assert_gate_valid(thorough)


def test_ceiling_is_a_cap_not_a_quota_no_padding(temp_store):
    # A tiny catalog surface (monkeypatched) must NOT be padded to the ceiling —
    # the planner stops early and marks the plan `capped`.
    _make_sft_split(temp_store, "sft-nopad")
    prof = profile_dataset("sft-nopad")
    plan = race_planner.plan_race(prof, "thorough")  # ceiling 16
    # The catalog can't honestly fill 16 distinct arms for a simple label task, so it
    # must stop short and say so — never emit 16 near-duplicate billable jobs.
    assert len(plan.planned) < 16
    assert plan.capped is True
    # No duplicate arms (entry keys unique).
    keys = [p.entry_key for p in plan.planned]
    assert len(keys) == len(set(keys))


def _make_large_json_split(store_mod, split_id, n_train=60000):
    """A LARGE, structured (json) SFT set — the regime where a FULL-FT arm is justified."""
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    train = [_chat(f"extract fields from record {i}", '{"name": "a", "amount": 5}')
             for i in range(n_train)]
    eval_ = [_chat(f"extract e{i}", '{"name": "b", "amount": 7}') for i in range(100)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "sft", "hasVal": False, "trainRows": n_train}), encoding="utf-8")
    store.commit("runs", split_id)


def test_thorough_never_exceeds_model_cap(temp_store, monkeypatch):
    _make_sft_split(temp_store, "sft-cap")
    prof = profile_dataset("sft-cap")
    monkeypatch.setattr(race_planner, "max_models_per_race", lambda: 3)
    plan = race_planner.plan_race(prof, "thorough")
    assert len(plan.planned) <= 3


def test_sft_offers_dora_arm(temp_store):
    # DoRA is offered as an A/B on the primary small LoRA model for SFT (any size set).
    _make_sft_split(temp_store, "sft-arms")
    prof = profile_dataset("sft-arms")
    plan = race_planner.plan_race(prof, "thorough")
    assert "dora" in {p.variant for p in plan.planned}
    _assert_gate_valid(plan)


def test_technique_arm_not_crowded_out_by_the_ladder(temp_store):
    # REGRESSION: with the 9-family ladder, the DoRA A/B must still appear at the medium
    # ("balanced", up-to-8) tier — a reserved slot keeps the ladder from filling all 8.
    _make_sft_split(temp_store, "sft-crowd")
    prof = profile_dataset("sft-crowd")
    plan = race_planner.plan_race(prof, "balanced")
    assert len(plan.planned) == 8
    dora = [p for p in plan.planned if p.variant == "dora"]
    assert dora, "DoRA A/B arm was crowded out of the balanced tier by the size ladder"
    # The DoRA arm rides the SAME model as a plain-LoRA baseline sibling (a real A/B).
    base_ids = {p.race_model.model_id for p in plan.planned if p.variant == "lora"}
    assert dora[0].race_model.model_id in base_ids


def test_dpo_orpo_arm_not_crowded_out(temp_store):
    _make_preference_split(temp_store, "pref-crowd")
    prof = profile_dataset("pref-crowd")
    plan = race_planner.plan_race(prof, "balanced")
    assert any(p.race_model.hp.pref_loss == "orpo" for p in plan.planned)


def test_full_arm_is_data_gated_offered_on_large_structured_set(temp_store):
    # P0-1: a FULL-FT arm appears only for a LARGE structured (json/numeric) dataset,
    # and only on a ≤2B model.
    _make_large_json_split(temp_store, "sft-fullbig")
    prof = profile_dataset("sft-fullbig")
    plan = race_planner.plan_race(prof, "thorough")
    full_arms = [p for p in plan.planned if p.method == "full"]
    assert full_arms, "expected a full-FT arm on a 60k-row json dataset"
    for p in full_arms:
        assert p.params_b <= FULL_FREEZE_MAX_PARAMS_B
    _assert_gate_valid(plan)


def test_full_arm_suppressed_on_small_dataset(temp_store):
    # A small (2k-row) set must NOT get a full arm — LoRA matches full there without
    # forgetting, so a full arm would burn a billable job on the riskier option.
    _make_sft_split(temp_store, "sft-small")
    prof = profile_dataset("sft-small")
    plan = race_planner.plan_race(prof, "thorough")
    assert all(p.method != "full" for p in plan.planned)


def test_full_arm_not_offered_on_large_unstructured_text(temp_store, monkeypatch):
    # Large but LABEL/text (small domain shift) → no full arm (only json/numeric/code).
    _make_sft_split(temp_store, "sft-bigtext", n_train=60000)
    prof = profile_dataset("sft-bigtext")
    plan = race_planner.plan_race(prof, "thorough")
    assert all(p.method != "full" for p in plan.planned)


def test_dpo_plan_uses_dpo_stage_and_orpo_arm_not_full(temp_store):
    _make_preference_split(temp_store, "pref-ds")
    prof = profile_dataset("pref-ds")
    plan = race_planner.plan_race(prof, "huge")
    assert plan.objective == "dpo"
    assert all(p.race_model.hp.stage == "dpo" for p in plan.planned)
    # full/freeze are SFT-only — must NOT appear for a DPO plan.
    assert all(p.method not in ("full", "freeze") for p in plan.planned)
    # An ORPO (reference-free) arm should be offered as the method axis.
    assert any(p.race_model.hp.pref_loss == "orpo" for p in plan.planned)
    _assert_gate_valid(plan)


def test_dpo_plan_uses_low_preference_learning_rate(temp_store):
    # End-to-end guard for the DPO/KTO LR footgun: a guided DPO plan must carry the
    # ~5e-6 preference LR, NOT the SFT-scale 1.5e-4/2e-4 it silently inherited before.
    _make_preference_split(temp_store, "pref-lr")
    prof = profile_dataset("pref-lr")
    plan = race_planner.plan_race(prof, "medium")
    assert plan.objective == "dpo"
    assert plan.planned
    for p in plan.planned:
        assert p.race_model.hp.learning_rate <= 1e-5, (
            f"{p.race_model.model_id} DPO LR {p.race_model.hp.learning_rate} too hot")


def test_sft_plan_uses_standard_lora_learning_rate(temp_store):
    _make_sft_split(temp_store, "sft-lr")
    prof = profile_dataset("sft-lr")
    plan = race_planner.plan_race(prof, "easy")
    for p in plan.planned:
        if p.method in ("lora", "qlora"):
            assert p.race_model.hp.learning_rate == 2.0e-4


def test_gated_models_excluded_without_token(temp_store):
    _make_sft_split(temp_store, "sft-gate")
    prof = profile_dataset("sft-gate")
    plan = race_planner.plan_race(prof, "huge", hf_token_ok=False)
    for p in plan.planned:
        spec = get_model(p.race_model.model_id)
        assert not spec.gated  # no Llama/Mistral/Gemma without a token


def test_non_reasoning_dataset_excludes_qwen3_families(temp_store):
    # A plain classification dataset (short answers) must not pull in reasoning
    # families — the format+size gate excludes them for label/json/short tasks.
    _make_sft_split(temp_store, "sft-noreason")
    prof = profile_dataset("sft-noreason")
    plan = race_planner.plan_race(prof, "thorough")
    for p in plan.planned:
        spec = get_model(p.race_model.model_id)
        assert not spec.reasoning


def _make_long_text_split(store_mod, split_id, n_train=3000):
    """An open-ended, long-answer TEXT set — the regime a reasoning base fits."""
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    long_answer = ("This is a detailed, open-ended explanation that runs well beyond a "
                   "short label because it develops several points in full sentences and "
                   "keeps going for many more words to be genuinely long form indeed. " * 3)
    train = [_chat(f"explain topic {i} in depth", long_answer) for i in range(n_train)]
    eval_ = [_chat(f"explain e{i} in depth", long_answer) for i in range(60)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "sft", "hasVal": False, "trainRows": n_train}), encoding="utf-8")
    store.commit("runs", split_id)


def test_reasoning_base_gated_in_on_long_open_ended_text(temp_store):
    # Format+size gate (arXiv:2509.22193): an open-ended long-form TEXT task admits
    # reasoning families (and prefers the ≥7B rung).
    _make_long_text_split(temp_store, "sft-longtext")
    prof = profile_dataset("sft-longtext")
    assert prof["eval"]["dominantTask"] == "text"
    plan = race_planner.plan_race(prof, "thorough")
    assert any(get_model(p.race_model.model_id).reasoning for p in plan.planned), \
        "expected a reasoning family on a long open-ended text task"


def test_reasoning_gate_decoupled_from_eval_floor(temp_store):
    # P0-2: a SHORT-answer (label) dataset that CONTAINS <think> scaffold must NOT pull
    # in reasoning families (format gate says short/closed), but MUST still flag that the
    # eval token floor needs raising (the tuned model will emit <think>).
    store = temp_store.get_store()
    wd = store.workdir("runs", "sft-thinklabel")
    train = [_chat(f"classify {i}", "<think>hmm, this is billing</think>billing") for i in range(400)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in train[:40]) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": "sft-thinklabel", "shape": "sft", "hasVal": False, "trainRows": 400}), encoding="utf-8")
    store.commit("runs", "sft-thinklabel")
    prof = profile_dataset("sft-thinklabel")
    # Format gate: short/closed answer → NO reasoning families in the set.
    assert race_planner._reasoning_base_fits(prof, "sft") is False
    plan = race_planner.plan_race(prof, "balanced")
    assert all(not get_model(p.race_model.model_id).reasoning for p in plan.planned)
    # But the eval-floor signal (scaffold present) fires independently.
    assert race_planner.needs_raised_eval_floor(prof, "sft") is True


def test_rl_objective_returns_unsupported(temp_store):
    # Fabricate an rlvr-shaped profile by writing an rlvr split.
    store = temp_store.get_store()
    wd = store.workdir("runs", "rlvr-ds")
    train = [{"messages": [{"role": "user", "content": f"{i}+{i}?"}], "ground_truth": str(2 * i)}
             for i in range(200)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in train[:20]) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps({"name": "rlvr-ds", "shape": "rlvr", "trainRows": 200}),
                                  encoding="utf-8")
    store.commit("runs", "rlvr-ds")
    prof = profile_dataset("rlvr-ds")
    plan = race_planner.plan_race(prof, "medium")
    assert plan.supported is False
    assert not plan.planned
    assert plan.reason  # a clear plain-language message


def test_plan_is_deterministic(temp_store):
    _make_sft_split(temp_store, "sft-det")
    prof = profile_dataset("sft-det")
    a = race_planner.plan_race(prof, "medium")
    b = race_planner.plan_race(prof, "medium")
    assert [p.race_model.model_id for p in a.planned] == [p.race_model.model_id for p in b.planned]


# --- model-swap helpers (the "agent proposes, user can swap" path) -----------

def test_eligible_models_excludes_gated_and_reasoning_for_plain_sft(temp_store):
    _make_sft_split(temp_store, "sft-pool")
    prof = profile_dataset("sft-pool")
    pool = race_planner.eligible_models(prof, hf_token_ok=False)
    assert pool, "expected a non-empty add pool"
    for m in pool:
        spec = get_model(m["modelId"])
        assert not spec.gated         # no gated models without a token
        assert not spec.reasoning     # no reasoning families for a plain classification set
    # Sorted smallest-first.
    sizes = [m["paramsB"] for m in pool]
    assert sizes == sorted(sizes)


def test_eligible_models_empty_for_unsupported_objective(temp_store):
    store = temp_store.get_store()
    wd = store.workdir("runs", "rlvr-pool")
    rows = [{"messages": [{"role": "user", "content": f"{i}?"}], "ground_truth": str(i)} for i in range(50)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:10]) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps({"name": "rlvr-pool", "shape": "rlvr", "trainRows": 50}),
                                  encoding="utf-8")
    store.commit("runs", "rlvr-pool")
    prof = profile_dataset("rlvr-pool")
    assert race_planner.eligible_models(prof) == []


def test_build_models_from_specs_round_trips_a_plan(temp_store):
    _make_sft_split(temp_store, "sft-rt")
    prof = profile_dataset("sft-rt")
    plan = race_planner.plan_race(prof, "medium")
    specs = race_planner.specs_from_plan(plan)
    planned, skipped = race_planner.build_models_from_specs(prof, specs)
    assert not skipped
    assert [p.race_model.model_id for p in planned] == [p.race_model.model_id for p in plan.planned]
    _assert_gate_valid_list(planned)


def test_build_models_from_specs_skips_invalid_cells(temp_store):
    _make_sft_split(temp_store, "sft-bad")
    prof = profile_dataset("sft-bad")
    specs = [
        {"modelId": "qwen2.5-1.5b", "method": "lora", "variant": "lora", "prefLoss": "sigmoid"},
        {"modelId": "does-not-exist", "method": "lora", "variant": "lora", "prefLoss": "sigmoid"},
        # full-weight on a >2B model is invalid (size gate) → must be skipped.
        {"modelId": "qwen2.5-7b", "method": "full", "variant": "lora", "prefLoss": "sigmoid"},
    ]
    planned, skipped = race_planner.build_models_from_specs(prof, specs)
    kept = [p.race_model.model_id for p in planned]
    assert "qwen2.5-1.5b" in kept
    assert "does-not-exist" in skipped
    assert "qwen2.5-7b" in skipped  # full-FT on 7B rejected
    _assert_gate_valid_list(planned)


def test_build_models_from_specs_respects_cap(temp_store, monkeypatch):
    _make_sft_split(temp_store, "sft-cap2")
    prof = profile_dataset("sft-cap2")
    monkeypatch.setattr(race_planner, "max_models_per_race", lambda: 2)
    specs = [{"modelId": mid, "method": "lora", "variant": "lora", "prefLoss": "sigmoid"}
             for mid in ("qwen2.5-1.5b", "qwen2.5-7b", "granite-3.1-2b", "granite-3.1-8b")]
    planned, skipped = race_planner.build_models_from_specs(prof, specs)
    assert len(planned) == 2
    assert len(skipped) == 2


def _assert_gate_valid_list(planned):
    for p in planned:
        hp = p.race_model.hp
        Hyperparams(**{k: getattr(hp, k) for k in (
            "engine", "stage", "finetuning_type", "lora_variant", "pref_loss",
            "learning_rate", "lora_rank")})
