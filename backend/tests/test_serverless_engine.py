# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker serverless engine — pure logic + integration seams.

No AWS: dataset reshaping, hyperparameter mapping, job naming, ModelPackageGroup
naming, entry_key engine axis, the engine-aware eval bridge, and the launch
dispatch (with the SDK trainer mocked) are all exercised without a real call.
"""
import json
import types

import pytest

from app.catalog import Hyperparams, get_model


# --- dataset reshaping -----------------------------------------------------

def test_messages_to_prompt_completion_strings():
    from app.engines.serverless_data import messages_to_prompt_completion

    msgs = [
        {"role": "system", "content": "You are a triage bot."},
        {"role": "user", "content": "ticket text"},
        {"role": "assistant", "content": "urgency: high"},
    ]
    row = messages_to_prompt_completion(msgs)
    # Both must be plain STRINGS (recipe converter requires it — proven in spike).
    assert isinstance(row["prompt"], str) and isinstance(row["completion"], str)
    assert row["completion"] == "urgency: high"
    assert "triage bot" in row["prompt"] and "ticket text" in row["prompt"]
    assert "urgency: high" not in row["prompt"]  # final assistant is NOT in prompt


def test_ranking_to_dpo_strings():
    from app.engines.serverless_data import ranking_to_dpo

    row = {
        "messages": [{"role": "user", "content": "q"}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
    }
    out = ranking_to_dpo(row)
    assert out == {"prompt": "q", "chosen": "good", "rejected": "bad"}


def test_ranking_to_dpo_tolerates_bare_string_turns():
    from app.engines.serverless_data import ranking_to_dpo

    out = ranking_to_dpo({"messages": [{"role": "user", "content": "q"}],
                          "chosen": "good", "rejected": "bad"})
    assert out["chosen"] == "good" and out["rejected"] == "bad"


def test_reshape_rejects_bad_rows():
    """Hardening (from the adversarial review): malformed rows raise a clear
    DataConversionError instead of crashing cryptically or silently corrupting."""
    from app.engines.serverless_data import (
        DataConversionError, messages_to_prompt_completion, ranking_to_dpo,
    )

    # no assistant turn
    with pytest.raises(DataConversionError, match="no assistant turn"):
        messages_to_prompt_completion([{"role": "system", "content": "x"},
                                       {"role": "user", "content": "q"}])
    # empty messages
    with pytest.raises(DataConversionError):
        messages_to_prompt_completion([])
    # empty completion
    with pytest.raises(DataConversionError, match="empty content"):
        messages_to_prompt_completion([{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": "  "}])
    # DPO missing/empty chosen|rejected must NOT become "None"/"" — it errors
    with pytest.raises(DataConversionError, match="chosen"):
        ranking_to_dpo({"messages": [{"role": "user", "content": "q"}]})
    with pytest.raises(DataConversionError):
        ranking_to_dpo({"messages": [{"role": "user", "content": "q"}],
                        "chosen": {"role": "assistant"}, "rejected": "bad"})  # chosen has no content


def test_convert_file_reports_offending_row(tmp_path):
    from app.engines.serverless_data import convert_file, DataConversionError

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "a"}]}) + "\n"
        + json.dumps({"data": "no messages key"}) + "\n",  # bad row 2
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    with pytest.raises(DataConversionError, match="line 2"):
        convert_file(src, dst, "sft")


def test_convert_file_sft(tmp_path):
    from app.engines.serverless_data import convert_file

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "q2"},
                                   {"role": "assistant", "content": "a2"}]}) + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    n = convert_file(src, dst, "sft")
    assert n == 2
    rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert all(set(r) == {"prompt", "completion"} for r in rows)
    assert rows[0]["completion"] == "a1"


# --- RLVR VERL reshape (explicit ground_truth; prompt stays an ARRAY) -------

def test_rlvr_to_verl_shape():
    from app.engines.serverless_data import rlvr_to_verl

    row = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "2+2?"},
        ],
        "ground_truth": "4",
    }
    out = rlvr_to_verl(row, "7")
    # VERL: id + prompt-as-ARRAY (NOT a string) + reward_model.ground_truth.
    assert out["id"] == "7"
    assert isinstance(out["prompt"], list)
    assert out["prompt"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "2+2?"},
    ]
    assert out["reward_model"] == {"ground_truth": "4"}


def test_rlvr_to_verl_rejects_bad_rows():
    from app.engines.serverless_data import DataConversionError, rlvr_to_verl

    with pytest.raises(DataConversionError):
        rlvr_to_verl({"messages": [], "ground_truth": "4"}, "0")  # empty prompt
    with pytest.raises(DataConversionError):
        # missing ground_truth — RLVR's verifiable target is NOT derived from prompt
        rlvr_to_verl({"messages": [{"role": "user", "content": "q"}]}, "0")
    with pytest.raises(DataConversionError):
        # blank ground_truth
        rlvr_to_verl({"messages": [{"role": "user", "content": "q"}],
                      "ground_truth": "  "}, "0")


def test_convert_file_rlvr(tmp_path):
    from app.engines.serverless_data import convert_file

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q1"}], "ground_truth": "a1"}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "q2"}], "ground_truth": "a2"}) + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    n = convert_file(src, dst, "rlvr")
    assert n == 2
    rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert all(set(r) == {"id", "prompt", "reward_model"} for r in rows)
    assert rows[0]["reward_model"]["ground_truth"] == "a1"
    assert isinstance(rows[0]["prompt"], list)  # array, unlike SFT's string


# --- RLVR Hyperparams gating (stage + preset reward) -----------------------

def test_hyperparams_rlvr_requires_preset_reward():
    # rlvr without a preset reward is rejected (would launch a reward-less job)
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlvr")
    # an off-list reward is rejected
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                    preset_reward_function="not_a_preset")
    # a valid preset is accepted
    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k")
    assert hp.preset_reward_function == "gsm8k"


def test_hyperparams_rlvr_custom_reward():
    # a custom reward_function_id is a valid RLVR reward (no preset needed)
    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     reward_function_id="abc123")
    assert hp.reward_function_id == "abc123"
    assert hp.preset_reward_function == ""


def test_hyperparams_rlvr_rejects_both_or_neither_reward():
    # both preset AND custom → error
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                    preset_reward_function="gsm8k", reward_function_id="abc")
    # neither → error
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlvr")


def test_hyperparams_clears_reward_id_for_non_rlvr():
    hp = Hyperparams(engine="sagemaker_serverless", stage="sft",
                     reward_function_id="abc")
    assert hp.reward_function_id == ""


def test_hyperparams_rlvr_is_serverless_only():
    # LLaMA-Factory can't do RLVR — gated by the engine capability matrix
    with pytest.raises(ValueError):
        Hyperparams(engine="llama_factory", stage="rlvr",
                    preset_reward_function="gsm8k")


def test_hyperparams_clears_preset_reward_for_non_rlvr():
    # a stray preset_reward on an sft run is cleared so it can't leak into a spec
    hp = Hyperparams(engine="sagemaker_serverless", stage="sft",
                     preset_reward_function="gsm8k")
    assert hp.preset_reward_function == ""


def test_preset_reward_functions_are_numeric_builtins():
    # AWS removed preset_reward_function from the open-weight GRPO recipe (verified
    # against sagemaker-train 1.13.1 + the open-weight docs, 2026-06-29). Presets are
    # now reconstructed as auto-provisioned built-in reward FUNCTIONS. Only the
    # numeric-answer presets survive (gsm8k/prime_math → numeric_match); prime_code
    # was dropped (a pure-python reward Lambda can't run code against tests).
    from app.reward_functions import PRESET_BUILTIN_REWARDS

    assert set(Hyperparams.PRESET_REWARD_FUNCTIONS) == {"gsm8k", "prime_math"}
    assert PRESET_BUILTIN_REWARDS == {"gsm8k": "numeric_match", "prime_math": "numeric_match"}


# --- hyperparameter mapping (enum/range snapping) --------------------------

def test_map_hyperparameters_snaps_to_recipe_enums():
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="sft",
                     lora_rank=8, lora_alpha=None, learning_rate=5e-4,
                     per_device_train_batch_size=2, gradient_accumulation_steps=8,
                     num_train_epochs=2.0, cutoff_len=4096)
    out = map_hyperparameters(hp)
    assert out["lora_rank"] in (8, 16, 32, 64, 128)
    assert out["lora_alpha"] in (16, 32, 64, 128, 256)
    assert out["global_batch_size"] in (8, 16, 32, 64, 128, 256, 512, 1024)
    assert out["learning_rate"] <= 1e-4  # recipe caps lr at 1e-4
    assert out["max_epochs"] == 2
    assert out["dataset_max_len"] == 4096


def test_grpo_learning_rate_snapped_to_recipe_default():
    """GRPO (rlvr/rlaif) is RL, not supervised — an inherited SFT LR (>= 1e-4) is
    snapped DOWN to the managed GRPO recipe default (1e-5), and everything is
    range-clamped to the recipe's [1e-7, 1e-3]. Verified against the live
    verl-grpo-rlvr/-rlaif recipes (learning_rate default 1e-5, min 1e-7, max 1e-3)."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    for stage in ("rlvr", "rlaif"):
        # The SFT default 1e-4 is treated as inherited → snapped to the GRPO default.
        hp = Hyperparams(engine="sagemaker_serverless", stage=stage,
                         preset_reward_function="gsm8k" if stage == "rlvr" else "",
                         reward_function_id="" if stage == "rlvr" else "rp",
                         learning_rate=1e-4)
        assert map_hyperparameters(hp, train_rows=144)["learning_rate"] == 1e-5, stage
        # An even higher (SFT-scale) LR is also snapped to the GRPO default.
        hp.learning_rate = 5e-4
        assert map_hyperparameters(hp, train_rows=144)["learning_rate"] == 1e-5, stage
        # A deliberately-low GRPO LR is RESPECTED (only range-clamped).
        hp.learning_rate = 2e-6
        assert map_hyperparameters(hp, train_rows=144)["learning_rate"] == 2e-6, stage
        # Below the recipe floor → clamped up to the min.
        hp.learning_rate = 1e-9
        assert map_hyperparameters(hp, train_rows=144)["learning_rate"] == 1e-7, stage

    # SFT is unaffected (caps at 1e-4, no snap-to-1e-5).
    sft = Hyperparams(engine="sagemaker_serverless", stage="sft", learning_rate=5e-4)
    assert map_hyperparameters(sft)["learning_rate"] == 1e-4


def test_map_hyperparameters_lora_rank_snaps_nearest():
    from app.engines.sagemaker_serverless import map_hyperparameters, _snap

    assert _snap(10, (8, 16, 32)) == 8
    assert _snap(13, (8, 16, 32)) == 16
    # alpha defaults to 2*rank, snapped
    hp = Hyperparams(engine="sagemaker_serverless", lora_rank=64, lora_alpha=None)
    assert map_hyperparameters(hp)["lora_alpha"] == 128


# --- RLVR batch enum + dataset-size guard (the e2e 'dataloader empty' bug) ---

def test_rlvr_batch_uses_floor_128_enum():
    """RLVR/GRPO's global_batch_size enum is [128,256,512,1024] (floor 128), NOT
    the SFT enum (floor 8). The default per_device(1)*grad_accum(8)=8 must snap UP
    to 128 for RLVR — the previous code snapped to 8, an off-enum value the recipe
    silently dropped, so the default 128 ran against a too-small split and VERL
    aborted with 'Train dataloader is empty'."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k")
    out = map_hyperparameters(hp, train_rows=500)  # plenty of rows
    assert out["global_batch_size"] in (128, 256, 512, 1024)
    assert out["global_batch_size"] == 128  # default per-device*accum=8 → floor 128


def test_rlvr_batch_snaps_down_to_fit_dataset():
    """With more rows than the floor, the batch fits UNDER the dataset (snap DOWN
    to a valid enum) so the dataloader is never empty."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k",
                     per_device_train_batch_size=8, gradient_accumulation_steps=64)  # desired 512
    # only 300 effective rows → must snap DOWN to 256 (not up to 512, which overflows)
    out = map_hyperparameters(hp, train_rows=300)
    assert out["global_batch_size"] == 256


def test_rlvr_too_few_rows_raises_clear_error():
    """An RLVR dataset smaller than the batch floor (128) cannot run — the only
    fix is more examples. We raise a clear pre-launch error instead of letting the
    recipe default 128 produce an empty dataloader (the original e2e failure: 37
    train rows < 128)."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k")
    with pytest.raises(ValueError, match="at least 128 training examples"):
        map_hyperparameters(hp, train_rows=37)  # the exact e2e failure size


def test_rlvr_floor_message_states_raw_row_bar_when_carved():
    """When no val file is uploaded the recipe carves ~10% for validation, so the
    effective train count is int(raw*0.9). The error must state the RAW-row bar
    (ceil(128/0.9)=143), not the post-split number, so a user who uploaded exactly
    128 rows knows they actually need 143 (or to add a val file)."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k")
    # user uploaded 128 raw rows, recipe carves to int(128*0.9)=115 effective
    with pytest.raises(ValueError) as ei:
        map_hyperparameters(hp, train_rows=115, raw_train_rows=128)
    msg = str(ei.value)
    assert "143" in msg          # the real raw-row bar
    assert "128" in msg          # the raw rows the user provided
    assert "validation" in msg   # explains the carve-out + the val-file remedy


def test_sft_batch_can_fit_small_dataset():
    """SFT's enum floor is 8, so a small SFT split snaps down rather than erroring
    (only RLVR has the hard 128 floor)."""
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="sft",
                     per_device_train_batch_size=4, gradient_accumulation_steps=8)  # desired 32
    out = map_hyperparameters(hp, train_rows=20)  # 20 rows → snap down to 16
    assert out["global_batch_size"] == 16


def test_snap_down_helper():
    from app.engines.sagemaker_serverless import _snap_down

    assert _snap_down(100, (128, 256, 512)) == 128  # below all → smallest enum
    assert _snap_down(300, (128, 256, 512)) == 256   # largest <= 300
    assert _snap_down(512, (128, 256, 512)) == 512   # exact
    assert _snap_down(37, (8, 16, 32, 64)) == 32     # largest <= 37


# --- launcher: global_batch_size force-verify (fail-fast for RLVR) ----------

class _FakeHP:
    """Mimics a recipe hyperparameter object. `coerce` optionally forces a stored
    value different from what was set (simulating recipe coercion); `reject` raises
    on setattr (simulating a hard rejection)."""

    def __init__(self, coerce=None, reject=False):
        self._coerce, self._reject, self._gbs = coerce, reject, None

    def __setattr__(self, k, v):
        if k in ("_coerce", "_reject", "_gbs"):
            object.__setattr__(self, k, v)
            return
        if k == "global_batch_size":
            if self._reject:
                raise ValueError("recipe rejects this knob")
            object.__setattr__(self, "_gbs", self._coerce if self._coerce is not None else v)
            return
        object.__setattr__(self, k, v)

    def __getattr__(self, k):
        if k == "global_batch_size":
            return object.__getattribute__(self, "_gbs")
        raise AttributeError(k)


def test_apply_global_batch_size_records_when_stored():
    from app.engines.serverless_launcher import _apply_global_batch_size
    hp = _FakeHP()
    applied = {}
    _apply_global_batch_size(hp, {"global_batch_size": 128}, applied, "rlvr")
    assert applied["global_batch_size"] == 128
    assert hp.global_batch_size == 128


def test_apply_global_batch_size_rlvr_fails_fast_on_coercion():
    """If the recipe silently coerces the value back to its default, RLVR must
    raise BEFORE the billable launch (not silently launch with the wrong batch)."""
    from app.engines.serverless_launcher import _apply_global_batch_size
    hp = _FakeHP(coerce=128)  # we ask for 256, recipe keeps 128
    with pytest.raises(ValueError, match="did not accept global_batch_size"):
        _apply_global_batch_size(hp, {"global_batch_size": 256}, {}, "rlvr")


def test_apply_global_batch_size_rlvr_fails_fast_on_rejection():
    from app.engines.serverless_launcher import _apply_global_batch_size
    hp = _FakeHP(reject=True)
    with pytest.raises(ValueError, match="did not accept global_batch_size"):
        _apply_global_batch_size(hp, {"global_batch_size": 128}, {}, "rlvr")


def test_apply_global_batch_size_sft_coercion_is_advisory():
    """For SFT/DPO a coerced batch is non-fatal (just not recorded as applied) —
    only RLVR hard-fails (its enum floor 128 makes a wrong batch catastrophic)."""
    from app.engines.serverless_launcher import _apply_global_batch_size
    hp = _FakeHP(coerce=8)
    applied = {}
    _apply_global_batch_size(hp, {"global_batch_size": 16}, applied, "sft")  # no raise
    assert "global_batch_size" not in applied  # mismatch → not recorded


def test_apply_global_batch_size_reverifies_even_if_already_applied():
    """The generic _specs loop may put global_batch_size in `applied` WITHOUT
    verifying the recipe stored it. The force-verify must re-check from scratch and
    drop a stale 'applied' entry when the recipe actually coerced the value."""
    from app.engines.serverless_launcher import _apply_global_batch_size
    hp = _FakeHP(coerce=128)
    applied = {"global_batch_size": 256}  # stale, as if the generic loop recorded it
    with pytest.raises(ValueError):
        _apply_global_batch_size(hp, {"global_batch_size": 256}, applied, "rlvr")
    assert "global_batch_size" not in applied  # stale entry was cleared


# --- job + ModelPackageGroup naming ----------------------------------------

def test_serverless_job_name_marks_engine_and_fits_63():
    from app.engines.serverless_naming import serverless_job_name

    n = serverless_job_name("qwen3-4b", "abc123def456", "20260612-1", stage="sft")
    assert n.startswith("slm-qwen3-4b-serverless-")
    assert "abc123def456" in n
    assert len(n) <= 63
    # DPO marks the stage so it's a distinct job name.
    d = serverless_job_name("qwen3-4b", "abc123def456", "20260612-1", stage="dpo")
    assert "serverless-dpo" in d


def test_serverless_job_name_survives_sdk_suffix_and_truncation():
    """Regression: the V3 SDK builds f'{base}-{YYYYMMDDHHMMSS}' then truncates to
    63. A 62-char base left the cut on a hyphen → invalid trainingJobName (must end
    alphanumeric), which failed a real RLAIF launch. Our base must be ≤48 AND, after
    the SDK's suffix+truncate, the final name must satisfy SageMaker's regex while
    keeping the 12-hex split id (leaderboard grouping) + the stage marker.
    """
    import re

    from app.engines.serverless_naming import serverless_job_name

    sm_re = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")  # SageMaker's rule
    split_re = re.compile(r"slm-.+?-([0-9a-f]{12})")  # leaderboard._resolve_train_job

    def sdk_unique(base: str, ts: str = "20260617124230") -> str:
        # mirror sagemaker.train's _get_unique_name
        return f"{base.replace('_', '-')}-{ts}"[:63]

    cases = [
        # The exact failing case: qwen3-1.7b + rlaif + a long stamp.
        ("qwen3-1.7b", "ece6bedccd26", "20260617-124230-0", "rlaif"),
        ("qwen3-1.7b", "ece6bedccd26", "20260617-124230-0", "rlvr"),
        # A very long model id must still keep the split id + marker.
        ("deepseek-r1-distill-qwen-14b", "ece6bedccd26", "20260617-124230-12", "rlaif"),
    ]
    for model_id, split_id, stamp, stage in cases:
        base = serverless_job_name(model_id, split_id, stamp, stage=stage)
        assert len(base) <= 48, (base, len(base))
        assert not base.endswith("-")
        assert split_id in base, f"split id dropped: {base}"  # leaderboard grouping
        assert f"serverless-{stage}" in base  # stage marker preserved
        # After the SDK's suffix + 63-truncate, the name is still SageMaker-valid.
        final = sdk_unique(base)
        assert sm_re.match(final), f"SDK name invalid: {final}"
        assert not final.endswith("-")
        assert split_re.search(final), f"split id unparsable after SDK suffix: {final}"


def test_tenant_model_package_group_default(monkeypatch):
    from app.engines import serverless_naming as sn
    from app import tenancy

    monkeypatch.setattr(tenancy, "current_tenant", lambda: tenancy.DEFAULT_TENANT)
    assert sn.tenant_model_package_group() == "slm-platform-serverless"


def test_ensure_model_package_group_creates_when_missing():
    from app.engines.serverless_naming import ensure_model_package_group

    created = {}

    class _SM:
        def describe_model_package_group(self, ModelPackageGroupName):
            if ModelPackageGroupName not in created:
                raise RuntimeError("not found")
            return {"ModelPackageGroupName": ModelPackageGroupName}

        def create_model_package_group(self, ModelPackageGroupName, ModelPackageGroupDescription):
            created[ModelPackageGroupName] = True

    class _Boto:
        def client(self, svc):
            return _SM()

    ensure_model_package_group(_Boto(), "grp-x")
    assert created == {"grp-x": True}


# --- entry_key engine axis (distinct rows; back-compat) --------------------

def test_entry_key_engine_axis():
    from app.race import entry_key_for

    # back-compat: default engine + lora → bare key (unchanged)
    assert entry_key_for("m", {"finetuning_type": "lora", "engine": "llama_factory"}) == "m"
    assert entry_key_for("m", {}) == "m"
    # serverless → distinct key
    assert entry_key_for("m", {"engine": "sagemaker_serverless"}) == "m::sagemaker_serverless"
    # method + engine both non-default → both tokens, method first
    assert entry_key_for("m", {"finetuning_type": "qlora", "engine": "llama_factory"}) == "m::qlora"
    # back-compat: sft/dpo/kto add NO stage token (every pre-RLVR key unchanged)
    assert entry_key_for("m", {"engine": "sagemaker_serverless", "stage": "sft"}) == "m::sagemaker_serverless"
    assert entry_key_for("m", {"engine": "sagemaker_serverless", "stage": "dpo"}) == "m::sagemaker_serverless"
    # RLVR appends its own token so SFT vs RLVR (same model/engine) are distinct rows
    assert entry_key_for("m", {"engine": "sagemaker_serverless", "stage": "rlvr"}) == "m::sagemaker_serverless::rlvr"


# --- engine-aware eval bridge ----------------------------------------------

def test_launch_eval_job_serverless_points_at_hf_merged(monkeypatch):
    """A serverless eval must target <artifact>/checkpoints/hf_merged/ on OUR
    image; the default path is unchanged (artifact root + source image)."""
    from app import orchestrate as orch
    from app.catalog import DecodingParams

    captured = {}

    class _SM:
        def describe_training_job(self, TrainingJobName):
            return {
                "TrainingJobStatus": "Completed",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/jobs/J/output/model"},
                "AlgorithmSpecification": {"TrainingImage": "aws-managed-recipe-image"},
            }

    class _Boto:
        def client(self, svc):
            return _SM()

    class _Est:
        class latest_training_job:
            name = "slm-eval-x"

        def __init__(self, **kw):
            captured["image_uri"] = kw.get("image_uri")

        def fit(self, inputs, wait):
            captured["model_uri"] = inputs["model"]

    cfg = types.SimpleNamespace(
        region="us-east-1", bucket="b", role_arn="arn:role",
        image_uri="our-eval-image:0.9.4", profile=None,
    )
    monkeypatch.setattr(orch, "load_aws_config", lambda: cfg)
    monkeypatch.setattr(orch, "_session", lambda c: (object(), _Boto()))
    monkeypatch.setattr(orch, "split_dir", lambda s: __import__("pathlib").Path("/tmp"))
    monkeypatch.setattr(orch, "_upload_files", lambda sm, c, files, prefix: "s3://b/ds")
    monkeypatch.setattr(orch, "_jobs_key_prefix", lambda: "slm-platform/jobs")
    monkeypatch.setattr(orch, "_jobs_s3_base", lambda c: "s3://b/slm-platform/jobs")
    monkeypatch.setattr(orch, "Estimator", _Est)
    monkeypatch.setattr(orch, "TrainingInput", lambda uri, input_mode: uri)

    # serverless: hf_merged subdir + OUR image
    orch.launch_eval_job("J", "split", DecodingParams(), "stamp", engine="sagemaker_serverless")
    assert captured["model_uri"] == "s3://b/jobs/J/output/model/checkpoints/hf_merged/"
    assert captured["image_uri"] == "our-eval-image:0.9.4"

    # default (LF): artifact root + the source job's own image
    captured.clear()
    orch.launch_eval_job("J", "split", DecodingParams(), "stamp")
    assert captured["model_uri"] == "s3://b/jobs/J/output/model"
    assert captured["image_uri"] == "aws-managed-recipe-image"


# --- serverless launch dispatch (SDK trainer mocked) -----------------------

def test_serverless_launch_reshapes_and_dispatches(tmp_path, monkeypatch):
    """End-to-end of the engine launch with boto + the V3 subprocess mocked:
    confirms it reshapes data, uploads, ensures the MPG, builds the spec, and
    returns a job descriptor. The trainer itself runs in a subprocess (V3 SDK),
    which is mocked here via _launch_via_subprocess."""
    from app.engines.sagemaker_serverless import SagemakerServerlessEngine

    run = tmp_path / "split"
    run.mkdir()
    (run / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "urgency: low"}]}) + "\n",
        encoding="utf-8",
    )

    uploaded = {}
    captured_spec = {}

    cfg = types.SimpleNamespace(region="us-east-1", bucket="b",
                                role_arn="arn:role", profile=None)

    class _S3:
        def upload_file(self, local, bucket, key):
            uploaded[key] = local

    class _Boto:
        def client(self, svc):
            return _S3()

    def _fake_subprocess(spec):
        captured_spec.update(spec)
        return {"jobName": "slm-qwen3-4b-serverless-x", "appliedHyperparameters": spec["hyperparameters"]}

    # The engine builds its session through the shared wrapper (which is what
    # puts the solution's user-agent on the call), so that is the seam to stub.
    monkeypatch.setattr("app.aws_clients.get_session", lambda **kw: _Boto())
    monkeypatch.setattr("app.aws_config.load_aws_config", lambda: cfg)
    monkeypatch.setattr("app.storage.split_dir", lambda s: run)
    monkeypatch.setattr("app.engines.serverless_naming.jobs_key_prefix",
                        lambda: "slm-platform/jobs")
    monkeypatch.setattr("app.engines.serverless_naming.tenant_model_package_group",
                        lambda: "mpg")
    monkeypatch.setattr("app.engines.serverless_naming.ensure_model_package_group",
                        lambda boto, name: name)
    monkeypatch.setattr("app.engines.sagemaker_serverless._launch_via_subprocess",
                        _fake_subprocess)

    model = get_model("qwen3-4b")
    hp = Hyperparams(engine="sagemaker_serverless", stage="sft")
    out = SagemakerServerlessEngine().launch_training_job(
        model=model, split_id="split", hp=hp, instance_type="ml.g5.4xlarge",
        stamp="x", max_run_seconds=3600, use_spot=False, image_tag=None,
    )
    assert out["engine"] == "sagemaker_serverless"
    assert out["serverlessModelId"] == "huggingface-reasoning-qwen3-4b"
    assert out["jobName"] == "slm-qwen3-4b-serverless-x"
    # the subprocess spec carried the right model + role + a train S3 uri
    assert captured_spec["serverlessModelId"] == "huggingface-reasoning-qwen3-4b"
    assert captured_spec["role"] == "arn:role"
    assert captured_spec["trainUri"].endswith("/train.jsonl")
    assert captured_spec["stage"] == "sft"
    assert captured_spec["presetRewardFunction"] == ""  # sft → no reward
    # the reshaped train file was uploaded
    assert any(k.endswith("/train.jsonl") for k in uploaded)


def test_serverless_launch_rlvr_carries_preset_reward(tmp_path, monkeypatch):
    """An RLVR launch reshapes data to VERL and resolves a preset reward to an
    auto-provisioned built-in reward's Evaluator ARN (the AWS recipe no longer
    accepts preset_reward_function), threading that ARN through customRewardEvaluatorArn
    while leaving presetRewardFunction EMPTY in the recipe spec; the preset NAME is
    still echoed in the returned descriptor for display."""
    from app.engines.sagemaker_serverless import SagemakerServerlessEngine

    run = tmp_path / "split"
    run.mkdir()
    # rlvr-shaped train data: prompt-only messages + explicit ground_truth.
    # Need >= ~143 rows so the effective train split (0.9) clears the RLVR batch
    # floor of 128 — fewer would (correctly) raise the dataset-too-small guard.
    rows = [json.dumps({"messages": [{"role": "user", "content": f"{i}+2?"}],
                        "ground_truth": str(i + 2)}) for i in range(150)]
    (run / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    uploaded = {}
    captured_spec = {}
    cfg = types.SimpleNamespace(region="us-east-1", bucket="b",
                                role_arn="arn:role", profile=None)

    class _S3:
        def upload_file(self, local, bucket, key):
            uploaded[key] = local

    class _Boto:
        def client(self, svc):
            return _S3()

    def _fake_subprocess(spec):
        captured_spec.update(spec)
        return {"jobName": "slm-qwen3-4b-serverless-rlvr-x",
                "appliedHyperparameters": {}}

    # The preset is resolved to a built-in reward Evaluator ARN — mock that
    # provisioning so the test doesn't touch AWS (the resolution itself is exercised
    # by its own reward_functions test).
    resolved = {}

    def _fake_ensure(preset, stamp=""):
        resolved["preset"] = preset
        return "arn:aws:sagemaker:us-east-1:1:hub-content/Pub/JsonDoc/builtin-numeric-match/1"

    # The engine builds its session through the shared wrapper (which is what
    # puts the solution's user-agent on the call), so that is the seam to stub.
    monkeypatch.setattr("app.aws_clients.get_session", lambda **kw: _Boto())
    monkeypatch.setattr("app.aws_config.load_aws_config", lambda: cfg)
    monkeypatch.setattr("app.storage.split_dir", lambda s: run)
    monkeypatch.setattr("app.reward_functions.ensure_preset_reward_evaluator_arn", _fake_ensure)
    monkeypatch.setattr("app.engines.serverless_naming.jobs_key_prefix",
                        lambda: "slm-platform/jobs")
    monkeypatch.setattr("app.engines.serverless_naming.tenant_model_package_group",
                        lambda: "mpg")
    monkeypatch.setattr("app.engines.serverless_naming.ensure_model_package_group",
                        lambda boto, name: name)
    monkeypatch.setattr("app.engines.sagemaker_serverless._launch_via_subprocess",
                        _fake_subprocess)

    model = get_model("qwen3-4b")
    hp = Hyperparams(engine="sagemaker_serverless", stage="rlvr",
                     preset_reward_function="gsm8k")
    out = SagemakerServerlessEngine().launch_training_job(
        model=model, split_id="split", hp=hp, instance_type="ml.g5.4xlarge",
        stamp="x", max_run_seconds=3600, use_spot=False, image_tag=None,
    )
    assert captured_spec["stage"] == "rlvr"
    assert resolved["preset"] == "gsm8k"
    # The preset key is NOT sent to the recipe (it would be rejected); the reward is
    # carried by the built-in reward Evaluator ARN.
    assert captured_spec["presetRewardFunction"] == ""
    assert captured_spec["customRewardEvaluatorArn"].endswith("builtin-numeric-match/1")
    # The preset NAME is still echoed for display in the returned descriptor.
    assert out["presetRewardFunction"] == "gsm8k"
    assert out["customRewardEvaluatorArn"].endswith("builtin-numeric-match/1")
    # the RLVR batch must be a valid GRPO enum value (floor 128), threaded through
    # the spec so the launcher forces it past the _specs gate.
    assert captured_spec["hyperparameters"]["global_batch_size"] in (128, 256, 512, 1024)
    # data was reshaped to VERL (id/prompt/reward_model), not sft prompt/completion
    up_train = next(local for k, local in uploaded.items() if k.endswith("/train.jsonl"))
    verl_row = json.loads(open(up_train).read().splitlines()[0])
    assert set(verl_row) == {"id", "prompt", "reward_model"}
    assert verl_row["reward_model"]["ground_truth"] == "2"


def test_serverless_engine_rejects_rlvr_without_preset(tmp_path, monkeypatch):
    """Defense in depth: even if a malformed hp bypassed Hyperparams validation,
    the engine refuses to launch a reward-less RLVR job."""
    from app.engines.sagemaker_serverless import SagemakerServerlessEngine

    run = tmp_path / "split"
    run.mkdir()
    (run / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "a"}]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.aws_config.load_aws_config",
                        lambda: types.SimpleNamespace(region="us-east-1", bucket="b",
                                                      role_arn="arn:role", profile=None))
    monkeypatch.setattr("app.storage.split_dir", lambda s: run)
    model = get_model("qwen3-4b")
    # a SimpleNamespace hp that skips Hyperparams.__post_init__ validation
    bad_hp = types.SimpleNamespace(engine="sagemaker_serverless", stage="rlvr",
                                   preset_reward_function="")
    with pytest.raises(ValueError, match="preset_reward_function"):
        SagemakerServerlessEngine().launch_training_job(
            model=model, split_id="split", hp=bad_hp, instance_type="ml.g5.4xlarge",
            stamp="x", max_run_seconds=3600, use_spot=False, image_tag=None,
        )


def test_serverless_launch_dispatched_async_not_inline(temp_store, monkeypatch):
    """A serverless entry must be DISPATCHED to the worker (state=launching), NOT
    launched inline in the request (the slow launch would blow the 29s API limit
    and block the LLaMA-Factory entries). Then the worker's launch_serverless_entry
    fills in the job. LLaMA-Factory entries still launch inline."""
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    dispatched = []
    monkeypatch.setattr(rm, "dispatch_worker", lambda p: dispatched.append(p) or True)
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "b"})
    inline = []
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: inline.append(kw) or {"jobName": f"job-{kw['stamp']}"})

    models = [
        rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams()),  # LF → inline
        rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams(engine="sagemaker_serverless")),  # serverless → dispatched
    ]
    race = rm.start_race("split-x", models, DecodingParams(), "20260615-1")
    states = {e.entry_key: e.state for e in race.entries}
    assert states["qwen3-1.7b"] == rm.TRAINING            # LF launched inline
    assert states["qwen3-1.7b::sagemaker_serverless"] == rm.LAUNCHING  # serverless dispatched
    # exactly one dispatch, for the serverless entry; LF did NOT dispatch
    assert len(dispatched) == 1 and dispatched[0]["task"] == "serverless_launch"
    # only the LF entry launched inline (serverless did not block the request)
    assert all(k["model_id"] == "qwen3-1.7b" for k in inline) and len(inline) == 1

    # now the worker completes the serverless launch
    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: {"jobName": "serverless-job-1"})
    rm.launch_serverless_entry(race.race_id, "qwen3-1.7b::sagemaker_serverless", "20260615-1-1")
    race2 = rm._load(race.race_id)
    sv = rm._find_entry(race2, "qwen3-1.7b::sagemaker_serverless")
    assert sv.state == rm.TRAINING and sv.train_job == "serverless-job-1"


def test_launch_serverless_entry_no_double_launch(temp_store, monkeypatch):
    """A duplicate worker dispatch must not re-launch an already-launched entry."""
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "dispatch_worker", lambda p: True)
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "b"})
    race = rm.start_race("split-x", [rm.RaceModel(model_id="qwen3-1.7b",
                          hp=Hyperparams(engine="sagemaker_serverless"))], DecodingParams(), "s")
    calls = []
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: calls.append(1) or {"jobName": "j1"})
    rm.launch_serverless_entry(race.race_id, "qwen3-1.7b::sagemaker_serverless", "s-0")
    rm.launch_serverless_entry(race.race_id, "qwen3-1.7b::sagemaker_serverless", "s-0")  # duplicate
    assert len(calls) == 1  # second call is a no-op (state already TRAINING)


def test_serverless_python_required(monkeypatch):
    """The trainer runs under a V3-SDK interpreter (process isolation, since V3 is
    incompatible with the V2 SDK). Without SLM_SERVERLESS_PYTHON set, the engine
    raises a clear, actionable error rather than guessing."""
    from app.engines.sagemaker_serverless import _serverless_python

    monkeypatch.delenv("SLM_SERVERLESS_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="SLM_SERVERLESS_PYTHON"):
        _serverless_python()


def test_launch_via_subprocess_parses_result(monkeypatch):
    """The subprocess JSON contract: a {"jobName":...} line on stdout is parsed;
    an {"error":...} payload raises."""
    import subprocess

    import app.engines.sagemaker_serverless as mod

    monkeypatch.setenv("SLM_SERVERLESS_PYTHON", "/usr/bin/true")

    class _Proc:
        def __init__(self, stdout, rc=0, stderr=""):
            self.stdout, self.returncode, self.stderr = stdout, rc, stderr

    # success: ignores noise, parses the JSON line
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc('noise\n{"jobName": "J1"}\n'))
    assert mod._launch_via_subprocess({"x": 1})["jobName"] == "J1"

    # error payload → RuntimeError
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc('{"error": "boom"}', rc=1))
    with pytest.raises(RuntimeError, match="boom"):
        mod._launch_via_subprocess({"x": 1})

    # the register-evaluator op returns {evaluatorArn} (NOT jobName) — also valid.
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc('{"evaluatorArn": "arn:ev"}'))
    assert mod._launch_via_subprocess({"op": "register_evaluator"})["evaluatorArn"] == "arn:ev"

    # neither jobName nor evaluatorArn → RuntimeError
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc('{"weird": 1}'))
    with pytest.raises(RuntimeError, match="no jobName/evaluatorArn"):
        mod._launch_via_subprocess({"x": 1})


def test_serverless_verification_namespace(temp_store):
    """Verify-before-trust gains a 'serverless' namespace (distinct from the LF
    image tiers). Mapped models show it; unmapped don't; a success promotes it."""
    from app.verifications import VERIFIED, model_status_map, set_status

    # mapped model → serverless key present, untested until proven
    m = model_status_map("qwen3-4b")
    assert "serverless" in m and m["serverless"]["status"] == "untested"
    assert {"stable", "latest"} <= set(m)  # LF tiers still there, distinct
    # unmapped model → no serverless key
    assert "serverless" not in model_status_map("phi-3.5-mini")
    # a successful serverless run promotes the serverless namespace only
    set_status("qwen3-4b", "serverless", VERIFIED, method="lora", job_name="j")
    m2 = model_status_map("qwen3-4b")
    assert m2["serverless"]["status"] == "verified"
    assert m2["stable"]["status"] != "verified"  # LF proof is independent


def test_autopromote_uses_serverless_namespace(temp_store, monkeypatch):
    """A completed serverless race entry autopromotes under the 'serverless' tag,
    not the model's LLaMA-Factory image tier."""
    from app import race as rm
    from app.verifications import model_status_map

    monkeypatch.setattr(rm, "describe_job", lambda j: {"trainingEndTime": "2026-06-13"})
    entry = rm.RaceEntry(model_id="qwen3-4b", model_display="Qwen3 4B",
                         instance_type="ml.g5.4xlarge",
                         hp={"engine": "sagemaker_serverless", "finetuning_type": "lora"},
                         train_job="slm-qwen3-4b-serverless-x")
    race = rm.Race(race_id="r", split_id="s", stamp="t", decoding={}, entries=[entry])
    rm._autopromote(race, entry)
    m = model_status_map("qwen3-4b")
    assert m["serverless"]["status"] == "verified"
    # the LF tier was NOT promoted by a serverless run
    assert m["stable"]["status"] != "verified"


def test_serverless_launch_without_mapping_raises():
    from app.engines.sagemaker_serverless import SagemakerServerlessEngine

    model = get_model("phi-3.5-mini")  # no serverless_model_id
    hp = Hyperparams(engine="sagemaker_serverless", stage="sft")
    with pytest.raises(ValueError, match="no serverless_model_id"):
        SagemakerServerlessEngine().launch_training_job(
            model=model, split_id="s", hp=hp, instance_type="ml.g5.2xlarge",
            stamp="x", max_run_seconds=3600, use_spot=False, image_tag=None,
        )


# --- serverless cost basis (no instance-hour crash) ------------------------

def test_train_cost_serverless_no_crash():
    from app.leaderboard import _train_cost

    class _SM:
        def describe_training_job(self, TrainingJobName):
            return {"ServerlessJobConfig": {"X": 1}, "BillableTimeInSeconds": 0,
                    "TrainingTimeInSeconds": 668}

    monkeypatch_resolved = _train_cost.__globals__.get("_resolve_train_job")
    import app.leaderboard as lb
    orig = lb._resolve_train_job
    lb._resolve_train_job = lambda sm, j: j
    try:
        out = _train_cost(_SM(), "slm-qwen3-4b-serverless-x")
    finally:
        lb._resolve_train_job = orig
    assert out["trainServerless"] is True
    assert out["trainInstance"] == "serverless"
    assert out["trainCostUsd"] is None  # priced out-of-band
    assert out["trainDurationSeconds"] == 668


def test_launcher_has_no_hard_v2_session_import():
    """Regression for the prod failure 'No module named sagemaker.session': the
    launcher must NOT statically import sagemaker.session (absent in the V3 SDK).
    It uses a V3-first/V2-fallback import instead. Guard by scanning the source."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "app" / "engines" / "serverless_launcher.py"
    text = src.read_text()
    # the bare top-level import is the bug; the guarded fallback (inside try) is OK
    assert "from sagemaker.session import Session" in text  # only as the fallback
    assert "sagemaker.core.helper.session_helper import Session" in text  # V3 primary
    # the V3 import must come FIRST (primary), V2 second (fallback)
    assert text.index("session_helper import Session") < text.index("from sagemaker.session import Session")


def test_describe_job_serverless_no_resourceconfig(monkeypatch):
    """describe_job (called on EVERY job by reconcile) must not KeyError on a
    serverless job, which has no ResourceConfig. Regression for the 'ResourceConfig'
    failure seen when a serverless entry was polled."""
    from app import orchestrate as orch
    import types as _t

    class _SM:
        def describe_training_job(self, TrainingJobName):
            return {"TrainingJobStatus": "Completed", "ServerlessJobConfig": {"X": 1},
                    "BillableTimeInSeconds": 0}  # NO ResourceConfig

    monkeypatch.setattr(orch, "load_aws_config",
                        lambda: _t.SimpleNamespace(region="us-east-1", profile=None))
    monkeypatch.setattr(orch, "_session", lambda c: (None, _t.SimpleNamespace(client=lambda s: _SM())))
    out = orch.describe_job("slm-qwen3-1-7b-serverless-x")
    assert out["status"] == "Completed"
    assert out["instanceType"] == "serverless"  # not a KeyError


# --- RLAIF (RL from AI feedback): prompt-only data + reward-prompt + GRPO floor ---

def test_rlaif_to_verl_shape():
    from app.engines.serverless_data import rlaif_to_verl

    row = {"messages": [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "greet the team"},
    ]}
    out = rlaif_to_verl(row, "3")
    # prompt-only VERL: id + prompt ARRAY + a PLACEHOLDER reward_model (the recipe's
    # VERL converter requires the field; the judge supplies the real reward).
    assert out["id"] == "3"
    assert isinstance(out["prompt"], list)
    assert out["prompt"][1] == {"role": "user", "content": "greet the team"}
    assert out["reward_model"] == {"ground_truth": ""}


def test_rlaif_to_verl_rejects_empty_prompt():
    from app.engines.serverless_data import DataConversionError, rlaif_to_verl

    with pytest.raises(DataConversionError):
        rlaif_to_verl({"messages": []}, "0")
    with pytest.raises(DataConversionError):
        rlaif_to_verl({}, "0")


def test_convert_file_rlaif(tmp_path):
    from app.engines.serverless_data import convert_file

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"messages": [{"role": "user", "content": "p1"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "p2"}]}) + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    n = convert_file(src, dst, "rlaif")
    assert n == 2
    rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert all(set(r) == {"id", "prompt", "reward_model"} for r in rows)  # placeholder reward_model
    assert all(r["reward_model"] == {"ground_truth": ""} for r in rows)
    assert isinstance(rows[0]["prompt"], list)


def test_hyperparams_rlaif_requires_reward_prompt_id():
    # rlaif without a reward_function_id (the reward-prompt record) is rejected
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlaif")
    # a preset is NOT a valid RLAIF reward
    with pytest.raises(ValueError):
        Hyperparams(engine="sagemaker_serverless", stage="rlaif",
                    preset_reward_function="gsm8k")
    # a reward_function_id (pointing at a reward_prompt record) is accepted
    hp = Hyperparams(engine="sagemaker_serverless", stage="rlaif",
                     reward_function_id="rp-1", reward_model_id="nova-pro")
    assert hp.reward_function_id == "rp-1"
    assert hp.reward_model_id == "nova-pro"
    assert hp.preset_reward_function == ""


def test_hyperparams_rlaif_is_serverless_only():
    with pytest.raises(ValueError):
        Hyperparams(engine="llama_factory", stage="rlaif", reward_function_id="rp-1")


def test_hyperparams_clears_reward_model_id_for_non_rlaif():
    hp = Hyperparams(engine="sagemaker_serverless", stage="sft", reward_model_id="nova")
    assert hp.reward_model_id == ""


def test_rlaif_batch_uses_grpo_floor_128():
    # RLAIF is GRPO like RLVR → floor-128 enum, NOT the SFT floor of 8.
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlaif", reward_function_id="rp-1")
    out = map_hyperparameters(hp, train_rows=500)
    assert out["global_batch_size"] in (128, 256, 512, 1024)
    assert out["global_batch_size"] == 128


def test_rlaif_too_few_rows_raises():
    from app.engines.sagemaker_serverless import map_hyperparameters

    hp = Hyperparams(engine="sagemaker_serverless", stage="rlaif", reward_function_id="rp-1")
    with pytest.raises(ValueError, match="at least 128 training examples"):
        map_hyperparameters(hp, train_rows=37)


def test_entry_key_rlaif_token():
    from app.race import entry_key_for

    # rlaif appends its own stage token (distinct from rlvr + from plain serverless)
    k_rlaif = entry_key_for("qwen3-4b", {"engine": "sagemaker_serverless", "stage": "rlaif",
                                         "finetuning_type": "lora", "reward_function_id": "rp-1"})
    k_rlvr = entry_key_for("qwen3-4b", {"engine": "sagemaker_serverless", "stage": "rlvr"})
    assert k_rlaif == "qwen3-4b::sagemaker_serverless::rlaif"
    assert k_rlvr == "qwen3-4b::sagemaker_serverless::rlvr"
    assert k_rlaif != k_rlvr
    # back-compat: existing stages unchanged
    assert entry_key_for("m", {"engine": "llama_factory", "stage": "sft"}) == "m"
    assert entry_key_for("m", {"engine": "llama_factory", "finetuning_type": "qlora",
                               "stage": "sft"}) == "m::qlora"
