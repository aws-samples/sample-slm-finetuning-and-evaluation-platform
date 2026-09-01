# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Engine protocol + registry.

An Engine encapsulates "how a (model, hyperparams, split) becomes a launched
SageMaker training job, and how its produced model is later evaluated". Today
there is exactly one production engine — LLaMA-Factory — and it remains the
default. New engines slot in beside it, gated behind a feature flag.

The protocol is intentionally narrow: it wraps only the launch seam
(`launch_training_job`). Everything downstream (describe_job, fetch_metrics,
the race state machine) already works off the generic SageMaker TrainingJob and
needs no engine awareness for the default path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # avoid import cycles / heavy imports at module load
    from ..catalog import Hyperparams, ModelSpec

# The engine every existing model uses. A ModelSpec with no engine field, and
# every persisted artifact created before engines existed, resolves to this.
DEFAULT_ENGINE = "llama_factory"

# Built-in enabled-state for a non-default engine when the operator hasn't saved
# an explicit choice. The SageMaker serverless engine is verified end-to-end and
# ships ON; an operator turns it OFF (or any future engine on) via the Settings
# toggle, which persists to config.json — the single source of truth (no env var,
# no redeploy). Engines absent from this map default OFF (dark-launched).
_ENGINE_DEFAULT_ENABLED: dict[str, bool] = {
    "sagemaker_serverless": True,
}


class EngineNotEnabled(RuntimeError):
    """Raised when a non-default engine is requested but its feature flag is
    off. Callers (launch_training_job / list_models) treat this as "engine
    unavailable" — the default LLaMA-Factory path is never gated."""


class Engine(Protocol):
    """How a model trains. The default implementation wraps today's code."""

    name: str

    def launch_training_job(
        self,
        model: "ModelSpec",
        split_id: str,
        hp: "Hyperparams",
        instance_type: str,
        stamp: str,
        max_run_seconds: int,
        use_spot: bool,
        image_tag: str | None,
    ) -> dict[str, Any]:
        """Render + upload + launch (wait=False). Returns the job descriptor
        dict (must include at least 'jobName')."""
        ...


def _flag_enabled(engine_name: str) -> bool:
    """A non-default engine is enabled by its saved flag. The default engine is
    ALWAYS enabled. For non-default engines the saved config is the SINGLE source
    of truth:

      * config.json key enable<Engine> (saved via the Settings toggle): if present
        it decides, so an operator flips the engine on OR off at runtime with no
        redeploy.
      * else the built-in default in _ENGINE_DEFAULT_ENABLED (serverless ON, any
        other engine OFF).

    There is no environment override — the toggle always sticks."""
    if engine_name == DEFAULT_ENGINE:
        return True
    # config.json toggle (saved via Settings), e.g. {"enableSagemakerServerless": true}
    camel = "enable" + "".join(p.capitalize() for p in engine_name.split("_"))
    try:
        from ..aws_config import _saved

        saved = _saved()
        if camel in saved and saved[camel] is not None:
            return bool(saved[camel])  # explicit toggle wins (on OR off)
    except Exception:  # noqa: BLE001 — never let config lookup break a launch
        pass
    # No saved choice → the built-in default for this engine.
    return _ENGINE_DEFAULT_ENABLED.get(engine_name, False)


def get_engine(name: str | None) -> Engine:
    """Resolve an engine by name. None/empty → the default LLaMA-Factory engine.
    Raises EngineNotEnabled if a known non-default engine is requested while its
    flag is off, and ValueError for an unknown engine name."""
    engine_name = (name or DEFAULT_ENGINE).strip() or DEFAULT_ENGINE

    if engine_name == DEFAULT_ENGINE:
        from .llama_factory import LlamaFactoryEngine

        return LlamaFactoryEngine()

    if engine_name == "sagemaker_serverless":
        if not _flag_enabled(engine_name):
            raise EngineNotEnabled(
                f"engine {engine_name!r} is not enabled "
                f"(enable it via Settings → Training engines, which saves "
                f"enableSagemakerServerless to config)"
            )
        from .sagemaker_serverless import SagemakerServerlessEngine

        return SagemakerServerlessEngine()

    raise ValueError(f"unknown engine: {engine_name!r}")


def engine_enabled(name: str | None) -> bool:
    """True if the engine resolves AND is enabled (default always True). Used by
    list_models to hide dark-launched engines' models from the picker."""
    try:
        get_engine(name)
        return True
    except (EngineNotEnabled, ValueError):
        return False
