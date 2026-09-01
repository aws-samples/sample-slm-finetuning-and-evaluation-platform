# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Custom RLVR reward functions: author a Python snippet → a SageMaker Evaluator
the serverless RLVR trainer scores rollouts with.

This is the headline capability the LLaMA-Factory engine can't do: train a model
to maximize a VERIFIABLE reward the user defines, instead of imitating answers.

What a reward function is, concretely:
  * the user authors  `reward(response, ground_truth) -> float`  (0..1) in the UI;
  * we wrap it in a Lambda (reward_templates/handler.py implements AWS's batch
    scoring contract) with `scoring` (eval.py's extraction+metrics) importable, so
    a one-line `return scoring.score("token_f1", response, ground_truth)` reuses
    the exact metric the leaderboard ranks on;
  * the Lambda is registered as a SageMaker Evaluator (type RewardFunction); its
    hub-content ARN is passed to RLVRTrainer(custom_reward_function=...).

SAFETY. Two execution sites, two different exposures:

  * At TRAINING time the snippet runs in the user's own least-privilege reward
    Lambda (their data, their logic), and the handler wraps every per-row call so a
    raising/NaN reward scores 0.0 — a bad row can never crash the billable GRPO
    loop. validate_snippet runs before packaging, so the AST rules are the only
    guard there: that Lambda imports the snippet with ordinary Python semantics.
  * The DRY-RUN (`try_reward`, POST /api/reward-functions/try) runs in-process in
    the shared, privileged backend Lambda, whose role can read the Hugging Face
    token secret, read and write every tenant's S3 prefix, and create SageMaker
    training jobs. That path gets a restricted `__builtins__` AND an `__import__`
    that returns _SafeModule façades rather than real modules, because a real
    module is a doorway back into the interpreter: `collections._sys` is the `sys`
    module, so an import allowlist of nothing but "harmless" stdlib was escapable.

Neither layer makes executing user-authored Python in a privileged process safe by
construction. The stronger design is to run the dry-run in the same throwaway
least-privilege Lambda the deploy path already builds; this is defence in depth
until then. Heavy AWS calls (Lambda create, Evaluator.create) are kept behind small
functions so the registry/validation layer is unit-testable.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .reward_templates import scoring

_TEMPLATE_DIR = Path(__file__).resolve().parent / "reward_templates"
# Registry doc (GLOBAL root json) keyed by tenant → {rewardId: {...}} so reward
# functions are tenant-isolated even though the store's root json is global.
_REGISTRY_FILE = "reward_functions.json"


class RewardError(ValueError):
    """A reward function is invalid (bad snippet) or can't be built/registered."""


# --- snippet validation ---------------------------------------------------- #

# What a snippet may import, and WHICH NAMES of each module it may touch. `scoring`
# is our eval.py mirror; the rest are safe stdlib for parsing/measuring answers.
#
# The per-module name list is load-bearing, not documentation. A live module object
# carries private references straight back into the interpreter — `collections._sys`
# IS the `sys` module, so `collections._sys.modules["os"]` used to be a complete
# escape from an allowlist containing nothing but "harmless" stdlib. `_safe_import`
# therefore hands out a _SafeModule façade built from these names instead of the
# real module, so there is no edge to walk in the first place.
_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "scoring": ("score", "extract_answer", "METRIC_NAMES"),
    "json": ("loads", "dumps", "JSONDecodeError"),
    "re": ("compile", "escape", "findall", "finditer", "fullmatch", "match",
           "search", "split", "sub", "subn",
           "DOTALL", "IGNORECASE", "MULTILINE", "VERBOSE", "I", "M", "S", "X"),
    "math": ("ceil", "copysign", "e", "exp", "fabs", "floor", "fsum", "inf",
             "isclose", "isfinite", "isinf", "isnan", "log", "log2", "log10",
             "nan", "pi", "sqrt", "tau", "trunc"),
    "string": ("ascii_letters", "ascii_lowercase", "ascii_uppercase", "capwords",
               "digits", "hexdigits", "octdigits", "printable", "punctuation",
               "whitespace"),
    # namedtuple is deliberately absent: it builds a class via exec internally and
    # a reward function has no need for it.
    "collections": ("Counter", "OrderedDict", "defaultdict", "deque"),
}
_ALLOWED_IMPORTS = frozenset(_MODULE_EXPORTS)
# Builtins that have no business in a pure scoring function. Most are already
# absent from _SAFE_BUILTIN_NAMES; naming them here turns a confusing NameError at
# exec time into an actionable validation error at author time.
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "super", "memoryview", "breakpoint", "help",
}


def validate_snippet(snippet: str) -> None:
    """Parse-validate a user reward snippet. Must define `reward(response,
    ground_truth)`; may import only the allowlisted modules; must not use eval/
    exec/open/etc. or reference ANY private (`_`-prefixed) name or attribute.
    Raises RewardError with an actionable message.

    This is the FIRST of two layers, and it is the error-message layer: it rejects
    escapes at parse time so the author sees something actionable. The isolation
    itself is `_safe_import`'s _SafeModule façades plus the restricted
    `__builtins__` in `try_reward`, and at training time the per-row try/except and
    the reward Lambda's own least-privilege role. Do NOT rely on this AST pass
    alone as a security boundary — an AST blocklist is inherently bypassable, which
    is why the façades exist. It DOES carry real weight on the deploy path, though:
    build_lambda_zip validates before packaging, and the packaged Lambda imports
    the snippet with ordinary Python semantics and no sandbox."""
    if not snippet or not snippet.strip():
        raise RewardError("reward snippet is empty")
    try:
        tree = ast.parse(snippet)
    except SyntaxError as e:
        raise RewardError(f"snippet has a syntax error: {e.msg} (line {e.lineno})") from e

    has_reward = False
    for node in ast.walk(tree):
        # must define reward(response, ground_truth)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reward":
            args = [a.arg for a in node.args.args]
            if len(args) < 2:
                raise RewardError("reward() must take (response, ground_truth)")
            has_reward = True
        # restrict imports. Three separate things to reject: a module outside the
        # allowlist, a SUBMODULE of an allowlisted module (the allowlist covers
        # top-level names only, and _safe_import refuses dotted paths, so accepting
        # them here would let the two layers disagree), and any PRIVATE name pulled
        # in by a from-import — `from collections import _sys` hands the snippet the
        # live `sys` module under a name the attribute rule below never sees.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "." in alias.name:
                    raise RewardError(
                        f"import {alias.name!r} is not allowed; reward snippets may "
                        f"import only the top-level modules {sorted(_ALLOWED_IMPORTS)}"
                    )
                if alias.name not in _ALLOWED_IMPORTS:
                    raise RewardError(
                        f"import {alias.name!r} is not allowed; reward snippets may "
                        f"import only {sorted(_ALLOWED_IMPORTS)}"
                    )
                if (alias.asname or "").startswith("_"):
                    raise RewardError(
                        f"aliasing an import to the private name {alias.asname!r} is "
                        "not allowed in a reward snippet"
                    )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                raise RewardError("relative imports are not allowed in a reward snippet")
            if "." in module or module not in _ALLOWED_IMPORTS:
                raise RewardError(
                    f"import from {node.module!r} is not allowed; reward snippets may "
                    f"import only {sorted(_ALLOWED_IMPORTS)}"
                )
            for alias in node.names:
                if alias.name == "*":
                    raise RewardError("`import *` is not allowed in a reward snippet")
                for bound in (alias.name, alias.asname or ""):
                    if bound.startswith("_"):
                        raise RewardError(
                            f"importing the private name {bound!r} is not allowed in a "
                            "reward snippet (anything starting with '_' is rejected)"
                        )
                if alias.name not in _MODULE_EXPORTS[module]:
                    raise RewardError(
                        f"{module}.{alias.name} is not available to a reward snippet "
                        f"(allowed: {', '.join(sorted(_MODULE_EXPORTS[module]))})"
                    )
        # Forbid dangerous builtins AT THE CALL SITE. This is kept only because it
        # produces the clearest message for the obvious `eval(...)` case; it is NOT
        # the guard. A call-site check alone was bypassable by binding the builtin to
        # another name first — `e = eval` then `e("...")` never produces an ast.Call
        # whose func is a Name in _FORBIDDEN_CALLS, and the payload string is never
        # AST-visited at all. The allowlist over name LOADS below is what actually
        # closes that, for every binding route (assignment, list element, default
        # argument, comprehension, …) rather than one at a time.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise RewardError(f"{node.func.id}() is not allowed in a reward snippet")
        # Forbid EVERY private identifier, not just dunders — attribute access and
        # bare names alike. A single leading underscore is enough to escape: a
        # dunder-only rule let `collections._sys.modules["os"]` through, because
        # `_sys` is one underscore and `.modules`/`["os"]` are ordinary lookups.
        # Blocking the whole `_*` namespace closes that family in one rule
        # (`_sys`, `__class__`, `__globals__`, `__builtins__`, `_ModuleType`, …).
        # A scoring function has no legitimate need for a private name, so the
        # usability cost is a throwaway `_` variable.
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise RewardError(
                f"attribute access to the private name {node.attr!r} is not allowed "
                "in a reward snippet (anything starting with '_' is rejected)"
            )
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise RewardError(
                f"use of the private name {node.id!r} is not allowed in a reward "
                "snippet (anything starting with '_' is rejected)"
            )
        # `global`/`nonlocal` would let a snippet rebind a name the sandbox owns.
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise RewardError("global/nonlocal are not allowed in a reward snippet")

    # ALLOWLIST every free name the snippet reads. This is the real guard, and it
    # replaces a blocklist that could be sidestepped by renaming: `e = eval` binds a
    # dangerous builtin under a harmless name, and a call-site blocklist never sees
    # it. Instead of enumerating what is forbidden, enumerate what may be read — a
    # name the snippet itself binds, a safe builtin, or an allowlisted module — and
    # reject everything else. `eval`, `open`, `__import__`, `compile`, `breakpoint`
    # and every other builtin outside _SAFE_BUILTIN_NAMES fail here no matter how
    # they are reached, because reading the name at all is what gets rejected.
    bound = _bound_names(tree)
    allowed = bound | set(_SAFE_BUILTIN_NAMES) | set(_ALLOWED_IMPORTS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in allowed:
            raise RewardError(
                f"the name {node.id!r} is not available to a reward snippet. A snippet may "
                f"use its own variables, the allowlisted modules {sorted(_ALLOWED_IMPORTS)}, "
                "and a safe subset of builtins — anything else (eval, open, __import__, "
                "compile, …) is rejected however it is reached"
            )
    if not has_reward:
        raise RewardError("snippet must define a function `reward(response, ground_truth)`")


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the snippet itself binds, anywhere in it.

    Deliberately scope-INSENSITIVE: one flat set across the whole snippet. That is
    more permissive than Python's real scoping (a comprehension variable leaks into
    the set), but it can only ever let the snippet read a name IT defined, never a
    builtin it did not — which is the property the allowlist needs. Being lenient
    here also means the check never rejects legitimate scoring code for a scoping
    subtlety, so the security rule does not become the reason someone disables it."""
    names: set[str] = set()

    def add_target(t: ast.AST) -> None:
        # Assignment targets nest: `a, (b, *c) = …`, `obj.attr`, `d[k]`. Only bare
        # Names bind; Attribute/Subscript targets mutate something already bound.
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                add_target(el)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    def add_args(a: ast.arguments) -> None:
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                names.add(arg.arg)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            add_args(node.args)
        elif isinstance(node, ast.Lambda):
            add_args(node.args)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                # `import a.b` binds `a`; the import rules above already reject dots.
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            add_target(node.target)
        elif isinstance(node, ast.NamedExpr):  # walrus
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchAs) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


# --- restricted execution sandbox (dry-run) -------------------------------- #

# Builtins a pure scoring function legitimately needs. Everything capable of
# reaching the filesystem/network/interpreter — eval/exec/compile/open/input/
# __import__/globals/locals/vars/getattr/setattr/type/object/super/… — is
# DELIBERATELY omitted, so a snippet exec'd with this as its ONLY `__builtins__`
# cannot break out even if it slips past validate_snippet. `__import__` is
# replaced with `_safe_import` (allowlist-gated) so `import scoring` still works.
_SAFE_BUILTIN_NAMES = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hex", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    # exception types a reward may raise/catch
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "AttributeError", "StopIteration",
    "RuntimeError",
})


class _SafeModule:
    """A read-only façade exposing only the allowlisted names of one module.

    `_safe_import` returns these instead of real module objects. The distinction is
    the whole point: a live module is a doorway back into the interpreter, because
    modules keep private aliases to their own imports. `collections._sys` is the
    real `sys` module, so a snippet that got hold of the genuine `collections` could
    reach `sys.modules["os"]` and run anything — with an import allowlist that
    looked entirely harmless. A façade has no such edge.

    `__getattribute__` (not `__getattr__`) is overridden so that EVERY lookup goes
    through the allowlist, including dunders — there is no `__class__` to pivot off
    even if a snippet somehow defeats validate_snippet.
    """

    __slots__ = ("_name", "_exports")

    def __init__(self, name: str, exports: dict[str, Any]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_exports", exports)

    def __getattribute__(self, item: str) -> Any:
        exports = object.__getattribute__(self, "_exports")
        if item in exports:
            return exports[item]
        modname = object.__getattribute__(self, "_name")
        raise AttributeError(
            f"{modname}.{item} is not available to a reward snippet "
            f"(allowed: {', '.join(sorted(exports))})"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("modules are read-only inside a reward snippet")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("modules are read-only inside a reward snippet")


# Façades are immutable and their exports never change, so build each one once.
_safe_module_cache: dict[str, _SafeModule] = {}


def safe_module(root: str) -> _SafeModule:
    """The _SafeModule façade for an allowlisted module root. Raises ImportError for
    anything not in _MODULE_EXPORTS."""
    cached = _safe_module_cache.get(root)
    if cached is not None:
        return cached
    names = _MODULE_EXPORTS.get(root)
    if names is None:
        raise ImportError(f"import of {root!r} is not allowed in a reward snippet")
    if root == "scoring":
        source: Any = scoring  # our in-process eval.py mirror; no sys.modules touch
    else:
        import importlib

        source = importlib.import_module(root)
    proxy = _SafeModule(root, {n: getattr(source, n) for n in names if hasattr(source, n)})
    _safe_module_cache[root] = proxy
    return proxy


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    """The reward sandbox's `__import__`: only the allowlisted modules resolve, and
    they resolve to a _SafeModule façade rather than the real module, so `import
    collections` cannot become `collections._sys.modules["os"]`. Submodule paths are
    rejected outright — every allowlisted module is a single top-level name."""
    root = name.split(".")[0]
    if root != name:
        raise ImportError(
            f"import of the submodule {name!r} is not allowed in a reward snippet"
        )
    return safe_module(root)


def _reward_sandbox_globals() -> dict[str, Any]:
    """Fresh exec-globals whose `__builtins__` is the curated safe subset. With
    _safe_import's façades, this is the real trust boundary for the in-process
    dry-run — validate_snippet is the author-time error message, not the boundary."""
    import builtins as _b

    safe = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_b, n)}
    safe["__import__"] = _safe_import
    return {"__builtins__": safe}


# Which ground_truth task each PRESET reward can actually grade. gsm8k/prime_math
# check a numeric/extractable answer (both resolve to the numeric_match built-in).
# Used by reward_domain_warning to catch a billable GRPO run that would silently
# score ~0 (e.g. gsm8k pointed at prose). Advisory — never hard-blocks.
_PRESET_EXPECTED_TASK = {
    "gsm8k": {"numeric"},
    "prime_math": {"numeric"},
}
# Which ground_truth task a CUSTOM metric reward can meaningfully score.
_METRIC_EXPECTED_TASK = {
    "numeric_match": {"numeric"},
    "json_valid": {"json"},
    "label_accuracy": {"label", "numeric"},
}


def reward_domain_warning(
    *, preset: str = "", metric: str = "", ground_truth_task: str | None = None
) -> str | None:
    """Advisory check: does the chosen reward's domain match the dataset's detected
    ground_truth task? Returns a human-readable warning string on a likely
    mismatch (so the run isn't silently rewarded ~0), or None when it looks fine or
    can't be determined. Pure + unit-testable; the launch path surfaces it as a
    non-blocking warning (the user may have a delimiter the reward extracts)."""
    if not ground_truth_task:
        return None
    if preset:
        expected = _PRESET_EXPECTED_TASK.get(preset)
        if expected and ground_truth_task not in expected:
            return (
                f"The '{preset}' reward expects a numeric/extractable answer, but the "
                f"dataset's ground_truth looks like '{ground_truth_task}'. The run may score "
                "~0 every step. Use a custom reward whose logic matches your task, or a "
                "dataset whose ground_truth is the final number."
            )
        return None
    if metric:
        expected = _METRIC_EXPECTED_TASK.get(metric)
        if expected and ground_truth_task not in expected:
            return (
                f"The custom reward's metric '{metric}' can't meaningfully score a "
                f"'{ground_truth_task}' ground_truth — it would return ~0 every step. Pick a "
                "metric (or write reward logic) that matches your ground_truth shape."
            )
    return None


def try_reward(snippet: str, response: str, ground_truth: str) -> float:
    """Score ONE (response, ground_truth) pair with a reward snippet — IN-PROCESS,
    no AWS — so the user can preview what the deployed Lambda would return before
    launching a billable GRPO run. Mirrors reward_templates/handler._score_one
    EXACTLY (same extraction of the response, same NaN/inf guard + [0,1] clamp) so
    the dry-run number matches the real reward at training time.

    The snippet is validated AND exec'd inside a restricted-`__builtins__` sandbox
    (see _reward_sandbox_globals): only safe builtins and an allowlist-gated
    `__import__` are in scope, and that `__import__` yields _SafeModule façades
    rather than real modules, so a snippet cannot walk from an allowlisted module
    into the interpreter (the `collections._sys.modules` path). This runs IN the
    privileged backend Lambda — a higher trust boundary than the per-user reward
    Lambda at training time — whose role can read the Hugging Face token secret,
    read and write every tenant's S3 prefix, and create SageMaker training jobs.
    Treat it as hardened, not as a proven sandbox: executing user-authored Python
    in-process is a defence-in-depth posture, and the stronger design is to run the
    dry-run in the same throwaway least-privilege reward Lambda the deploy path
    already builds. `scoring` is bound directly and also importable via `import
    scoring`, matching the deployed handler where scoring.py is packaged flat.
    Raises RewardError on an invalid/raising snippet so the caller can surface an
    actionable message (the deployed handler would instead score 0.0 per-row, but
    for a dry-run the user wants to SEE the error)."""
    validate_snippet(snippet)
    ns: dict[str, Any] = _reward_sandbox_globals()
    # The façade, not the live module — binding the real `scoring` here would hand
    # back a module object and undo _safe_import's whole purpose.
    ns["scoring"] = safe_module("scoring")
    try:
        exec(compile(snippet, "<reward-snippet>", "exec"), ns)  # noqa: S102 — sandboxed builtins, validated snippet
    except Exception as e:  # noqa: BLE001
        raise RewardError(f"reward snippet failed to load: {e}") from e
    fn = ns.get("reward")
    if not callable(fn):
        raise RewardError("snippet must define a function `reward(response, ground_truth)`")
    try:
        val = float(fn(response or "", ground_truth or ""))
    except Exception as e:  # noqa: BLE001
        raise RewardError(f"reward() raised on this sample: {e}") from e
    if val != val or val in (float("inf"), float("-inf")):  # NaN/inf guard
        val = 0.0
    return max(0.0, min(1.0, val))


def _fill_reward_prompt(prompt_text: str, prompt: str, response: str) -> str:
    """Substitute {{prompt}} / {{response}} (whitespace-tolerant) in a judge rubric.
    Uses the SAME normalization regex as validate_reward_prompt so `{{ prompt }}`
    works identically."""
    out = re.sub(r"\{\{\s*prompt\s*\}\}", lambda _m: prompt or "", prompt_text)
    out = re.sub(r"\{\{\s*response\s*\}\}", lambda _m: response or "", out)
    return out


def try_reward_prompt(
    prompt_text: str,
    prompt: str,
    response: str,
    judge_model_id: str = "",
    *,
    _client: Any = None,
) -> dict[str, Any]:
    """Dry-run an RLAIF judge RUBRIC on one (prompt, response) pair — the judge
    analogue of try_reward, so a user can SEE whether a rubric discriminates good
    from bad BEFORE a billable GRPO run (RLAIF has no cheap small run — the GRPO
    batch floor is 128).

    Fills {{prompt}}/{{response}}, calls the judge via the model-agnostic Bedrock
    Converse API (same path as judge.py), parses the rubric's required
    {"score":0..1,"reasoning":...} JSON, applies the SAME NaN/inf guard + [0,1]
    clamp as try_reward, and NEVER raises on a malformed judge reply (mirrors the
    scoring.score never-raises contract — returns {"score":0.0,"error":...} so a
    calibration loop is not killed by one bad row).

    Validation that DOES raise (caller-actionable, before any billable Converse):
      * the rubric is missing a placeholder (validate_reward_prompt), or
      * judge_model_id is a non-empty id NOT in ALLOWED_JUDGE_MODELS.
    `_client` is for tests (inject a stub Converse client); prod builds one.
    Returns {"score": float in [0,1], "reasoning": str, "error": str|None}.
    """
    validate_reward_prompt(prompt_text)  # raises if placeholders missing
    jm = (judge_model_id or "").strip()
    if jm and jm not in ALLOWED_JUDGE_MODELS:
        raise RewardError(
            f"unknown judge model {jm!r}; choose one of {list(ALLOWED_JUDGE_MODELS)} "
            "(or leave blank for the recipe default)"
        )
    # Blank → the recipe default isn't invokable for an inline dry-run, so fall back
    # to the platform's default judge (Sonnet via Converse) just to PREVIEW the
    # rubric. The deployed reward still uses the chosen/recipe judge — this is an
    # indicative spread, surfaced as such in the UI.
    model_id = jm or _DRY_RUN_FALLBACK_JUDGE
    filled = _fill_reward_prompt(prompt_text, prompt, response)

    client = _client
    if client is None:
        from .aws_config import load_aws_config
        from .orchestrate import _session

        cfg = load_aws_config()
        _, boto_sess = _session(cfg)
        client = boto_sess.client("bedrock-runtime", region_name=cfg.region)

    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": filled}]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.0},
        )
        out_msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in out_msg.get("content", []))
    except Exception as e:  # noqa: BLE001 — a judge/Bedrock error must not kill the loop
        return {"score": 0.0, "reasoning": "", "error": f"judge call failed: {e}"}

    score, reasoning, err = _parse_judge_score(text)
    return {"score": score, "reasoning": reasoning, "error": err}


# The dry-run judge when the rubric's chosen judge is blank/recipe-default: the
# recipe default isn't Converse-invokable inline, so PREVIEW with the platform's
# default judge. Indicative only (the deployed reward uses the real recipe judge).
_DRY_RUN_FALLBACK_JUDGE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _parse_judge_score(text: str) -> tuple[float, str, str | None]:
    """Pull {"score":0..1,"reasoning":...} out of a judge reply. Tolerant of stray
    prose / fences. Returns (clamped_score, reasoning, error). Never raises — a
    malformed reply yields (0.0, "", "<reason>") so the calibration loop survives."""
    import math

    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return 0.0, "", "judge reply had no JSON object"
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return 0.0, "", "judge reply was not valid JSON"
    raw = obj.get("score")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0, str(obj.get("reasoning", "")), "judge reply had no numeric score"
    if math.isnan(val) or math.isinf(val):  # NaN/inf guard (mirrors try_reward)
        val = 0.0
    return max(0.0, min(1.0, val)), str(obj.get("reasoning", "")), None


# The reward↔leaderboard-metric loop: which LEADERBOARD rank metric (the metric
# the Investigate proposal recommends ranking on) can be turned into a verifiable
# RLVR reward — "train against the metric you ship on". A rank metric maps to a
# reward ONLY when it's a per-row VERIFIABLE check the reward Lambda can compute
# (scoring.METRIC_NAMES). Rank metrics that aren't verifiable per-row have NO
# automatic reward and return None (the UI then explains why):
#   * llm_judge:*        — an LLM judge is not a deterministic/verifiable check;
#                          RLVR needs a checkable target, not a model's opinion.
#   * json_structural /  — structural/key-level JSON scoring isn't a standalone
#     json_key_recall      reward scorer; json_valid is the closest verifiable
#                          check, but mapping silently would overclaim, so we
#                          leave it to the user (pick a preset / write custom).
def reward_metric_for_rank(rank_metric: str | None) -> str | None:
    """Map a leaderboard rank metric → the reward metric that mirrors it, or None
    when no verifiable per-row reward equals that metric. Pure + unit-testable;
    the single source of truth for the reward↔metric loop (UI never hardcodes it)."""
    if not rank_metric:
        return None
    # A rank metric IS rewardable iff it's also a verifiable per-row scorer. Keep
    # this an explicit membership check (not a guess) so it can never recommend a
    # reward the Lambda can't actually compute.
    return rank_metric if rank_metric in scoring.METRIC_NAMES else None


def metric_reward_snippet(metric: str) -> str:
    """A ready-made snippet for 'reward = the leaderboard metric' — the common
    case, so a user can pick a metric instead of writing code. Reuses scoring."""
    if metric not in scoring.METRIC_NAMES:
        raise RewardError(f"unknown metric {metric!r}; choose one of {list(scoring.METRIC_NAMES)}")
    return (
        "import scoring\n\n"
        "def reward(response, ground_truth):\n"
        f"    # reward = the {metric} the leaderboard ranks on\n"
        f"    return scoring.score({metric!r}, response, ground_truth)\n"
    )


# --- preset rewards → auto-provisioned built-in reward functions ------------ #

# AWS REMOVED preset RLVR rewards from the open-weight GRPO recipe. The current
# managed recipe / SDK RLVRTrainer takes ONLY a custom-reward Evaluator ARN and
# explicitly DELETES `preset_reward_function` from the recipe hyperparameters
# (verified against sagemaker-train 1.13.1 + the AWS open-weight customization
# docs, 2026-06-29). So a "preset" is now reconstructed as an AUTO-PROVISIONED
# built-in reward FUNCTION (a reward Lambda + Evaluator) — exactly the
# AWS-prescribed custom-reward pattern — whose Evaluator ARN is passed as
# custom_reward_function.
#
# Both surviving presets grade a numeric answer, so they map to the verifiable
# `numeric_match` scorer (extract the final number, exact compare — faithful to
# what gsm8k/prime_math actually check) and SHARE one built-in reward (keyed on the
# metric → one Lambda + one Evaluator, idempotent per tenant). `prime_code` is NOT
# offered as a preset: a pure-python reward Lambda cannot run code against tests, so
# a built-in would be a misleading label — author a custom reward with execution
# tooling for true test-passing instead.
PRESET_BUILTIN_REWARDS: dict[str, str] = {
    "gsm8k": "numeric_match",
    "prime_math": "numeric_match",
}
# Display name for a built-in reward, keyed by its metric (the record is shared
# across presets that map to the same metric, so the label is metric-based).
_BUILTIN_REWARD_LABEL = {
    "numeric_match": "Built-in numeric-answer reward (preset)",
}


def builtin_reward_id(metric: str) -> str:
    """Stable per-tenant id for the built-in reward that scores with `metric`. Keyed
    ONLY on the metric so presets sharing a metric (gsm8k/prime_math → numeric_match)
    reuse ONE registry record + ONE AWS Lambda/Evaluator (idempotent). The metric's
    underscore is hyphenated so the derived AWS Lambda + SageMaker Evaluator
    (hub-content) names match the proven hyphen-only convention the custom-reward
    path uses (avoids any underscore restriction on a hub-content name)."""
    return f"builtin-{metric.replace('_', '-')}"


def ensure_preset_reward_evaluator_arn(preset: str, stamp: str = "") -> str:
    """Resolve a preset reward fn → a DEPLOYED built-in reward function's Evaluator
    ARN, auto-provisioning the Lambda + Evaluator on first use (idempotent per
    tenant). This is how the serverless RLVR engine consumes a preset now that the
    AWS recipe no longer accepts `preset_reward_function` (it takes ONLY a
    custom-reward Evaluator ARN). Raises RewardError on an unknown preset or a deploy
    failure. Synchronous (build zip → Lambda → Evaluator) — called from the worker's
    off-request launch path, the same place the custom-reward path resolves its ARN."""
    metric = PRESET_BUILTIN_REWARDS.get(preset)
    if metric is None:
        raise RewardError(
            f"unknown preset reward function {preset!r}; expected one of "
            f"{tuple(PRESET_BUILTIN_REWARDS)}"
        )
    rid = builtin_reward_id(metric)
    rec = get_reward_function(rid)
    if rec and rec.get("evaluatorArn"):
        return rec["evaluatorArn"]  # already provisioned — reuse
    # CONCURRENCY NOTE: the registry is a single global last-writer-wins doc (no
    # locking — same property as everywhere else in the platform). If two sibling
    # RLVR entries in one race both hit a cold builtin reward, both may deploy. That
    # is SAFE for the launch: deploy_reward_function is idempotent on the AWS
    # resource NAME (keyed on lambda_hash → same Lambda + same Evaluator name), so
    # both resolve to the same reward, and each returns a valid ARN to its own
    # launch. The only cost is a redundant (idempotent) deploy + a possible transient
    # 'deploying' display state on the Reward-functions page; no reward-less or
    # wrong-reward job can result.
    # First use: create the registry record (stable id), then deploy it to AWS.
    snippet = metric_reward_snippet(metric)
    if rec is None:
        save_reward_function(RewardFunction(
            id=rid, name=_BUILTIN_REWARD_LABEL.get(metric, f"Built-in {metric} reward"),
            snippet=snippet, kind="metric", metric=metric,
            lambda_hash=reward_hash(snippet), created_stamp=stamp,
        ))
        rec = get_reward_function(rid) or {}
    from .reward_deploy import deploy_reward_function

    snippet = rec.get("snippet") or snippet
    lambda_key = rec.get("lambdaHash") or reward_hash(snippet)
    _set_status(rid, status="deploying", error="")
    try:
        out = deploy_reward_function(
            rid, rec.get("name") or _BUILTIN_REWARD_LABEL.get(metric, "Built-in reward"),
            build_lambda_zip(snippet), lambda_key=lambda_key)
    except Exception as e:  # noqa: BLE001
        # A sibling launch (same preset, concurrent worker in this race) may have
        # provisioned it between our checks — reuse its ARN rather than failing.
        fresh = get_reward_function(rid)
        if fresh and fresh.get("evaluatorArn"):
            return fresh["evaluatorArn"]
        _set_status(rid, status="failed", error=str(e))
        raise RewardError(
            f"failed to provision the built-in reward for preset {preset!r}: {e}"
        ) from e
    _set_status(rid, status="deployed", error="",
                lambdaArn=out["lambdaArn"], evaluatorArn=out["evaluatorArn"])
    return out["evaluatorArn"]


# --- RLAIF reward-prompt validation (no code; an AI-judge prompt) ----------- #

# The two placeholders the RLAIF judge fills with the rollout's prompt + the
# model's response. Without them the judge can't see what it's scoring.
_REWARD_PROMPT_PLACEHOLDERS = ("{{prompt}}", "{{response}}")

# Judge models the RLAIF recipe accepts as reward_model_id (the V3 SDK's
# rlaif_trainer._ALLOWED_REWARD_MODEL_IDS, filtered to the regions we deploy in —
# us-east-1). A real launch REJECTS anything else (an Amazon Nova id is NOT valid,
# which only a real run surfaces). "" is allowed = the recipe's default judge.
# Keep in sync with the SDK; validated in make_reward_prompt so a bad judge can't
# reach a billable launch.
ALLOWED_JUDGE_MODELS = (
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
    "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-coder-30b-a3b-v1:0",
)


def validate_reward_prompt(prompt: str) -> None:
    """Validate an RLAIF reward (judge) prompt. It must be non-empty and contain
    BOTH the {{prompt}} and {{response}} placeholders (the judge fills them with
    the rollout's prompt + the model's response). Advisory beyond that — prompt
    QUALITY can't be checked statically; we only enforce the load-bearing pieces.
    Raises RewardError with an actionable message. Whitespace inside the braces
    (e.g. `{{ prompt }}`) is tolerated."""
    if not prompt or not prompt.strip():
        raise RewardError("reward prompt is empty")
    # Normalize `{{ prompt }}` → `{{prompt}}` for the membership check.
    compact = re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{{\1}}", prompt)
    missing = [p for p in _REWARD_PROMPT_PLACEHOLDERS if p not in compact]
    if missing:
        raise RewardError(
            "reward prompt must include the "
            + " and ".join(missing)
            + " placeholder(s) so the AI judge can see the rollout's prompt and the "
            "model's response. Example: 'Rate the response to {{prompt}}: {{response}} "
            "— reply with JSON {\"score\": 0..1, \"reasoning\": \"...\"}'."
        )


# --- Lambda packaging (pure: snippet → zip bytes) --------------------------- #


def build_lambda_zip(snippet: str) -> bytes:
    """Package the reward Lambda: handler.py + scoring.py + the user's snippet as
    user_reward.py. Pure (no AWS) so it's unit-testable. Validates first."""
    validate_snippet(snippet)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("handler.py", (_TEMPLATE_DIR / "handler.py").read_text(encoding="utf-8"))
        zf.writestr("scoring.py", (_TEMPLATE_DIR / "scoring.py").read_text(encoding="utf-8"))
        zf.writestr("user_reward.py", snippet)
    return buf.getvalue()


def reward_hash(snippet: str) -> str:
    """Stable 12-hex hash of a reward snippet — used for the AWS Lambda NAME so
    identical snippets reuse the same Lambda/Evaluator (idempotent create), and
    re-launching a race doesn't pile up duplicate AWS resources. This is NOT the
    record id (two differently-named rewards may share a snippet — e.g. two metric
    rewards on the same metric — and must stay DISTINCT registry rows)."""
    return hashlib.sha256(snippet.strip().encode("utf-8")).hexdigest()[:12]


def _name_slug(name: str) -> str:
    """A short filesystem/identifier-safe slug from a reward name (for the id)."""
    safe = "".join(c if c.isalnum() else "-" for c in name.strip().lower())
    safe = "-".join(p for p in safe.split("-") if p)  # collapse runs of '-'
    return (safe[:24] or "reward")


def reward_id(name: str, snippet: str, stamp: str) -> str:
    """The UNIQUE registry id for a reward function record. Combines a name slug,
    the snippet hash, and the creation stamp so two user-distinct rewards never
    collide (the snippet-only hash did — a second metric reward silently
    overwrote the first and became un-deletable as a distinct row)."""
    base = f"{_name_slug(name)}-{reward_hash(snippet)}"
    # The stamp keeps same-name+same-snippet re-creates distinct; fall back to the
    # base alone when no stamp (deterministic for tests that omit it).
    return f"{base}-{stamp}"[:60] if stamp else base


# --- registry (per-tenant, persisted) -------------------------------------- #


@dataclass
class RewardFunction:
    id: str  # UNIQUE per record — reward_id(name, snippet, stamp), NOT the snippet hash
    name: str
    snippet: str
    # "snippet" (user code) | "metric" (generated from a metric) — both VERIFIABLE
    # RLVR rewards packaged as a Lambda. "reward_prompt" — a NON-verifiable RLAIF
    # AI-judge prompt (no Lambda; the prompt text is passed inline to RLAIFTrainer).
    kind: str = "snippet"
    metric: str | None = None  # set when kind == "metric"
    # RLAIF reward-prompt fields (kind == "reward_prompt"): the judge prompt text
    # (with {{prompt}}/{{response}} placeholders) and the optional judge model id.
    # The prompt is the artifact — there is no Lambda/Evaluator to build, so a
    # reward_prompt is usable ("deployed") the moment it's validated + saved.
    prompt: str = ""
    reward_model_id: str = ""  # judge model the recipe scores with ("" = recipe default)
    prompt_s3_uri: str = ""  # reward_prompt: where the prompt was uploaded (Evaluator source)
    lambda_hash: str = ""  # reward_hash(snippet|prompt) — the AWS resource NAME (idempotent reuse)
    lambda_arn: str = ""  # filled once deployed to AWS
    evaluator_arn: str = ""  # the SageMaker Evaluator hub-content ARN
    status: str = "draft"  # draft | deploying | deployed | failed
    error: str = ""  # deploy failure message (when status == failed)
    created_stamp: str = ""  # caller-supplied (no time in lib code)

    def to_dict(self) -> dict[str, Any]:
        # A reward_prompt is "deployed" once its REWARD_PROMPT Evaluator is
        # registered (no Lambda); a verifiable reward needs both its Lambda +
        # Evaluator ARNs.
        deployed = (
            bool(self.evaluator_arn) if self.kind == "reward_prompt"
            else bool(self.lambda_arn and self.evaluator_arn)
        )
        return {
            "id": self.id, "name": self.name, "snippet": self.snippet,
            "kind": self.kind, "metric": self.metric,
            "prompt": self.prompt, "rewardModelId": self.reward_model_id,
            "promptS3Uri": self.prompt_s3_uri,
            "lambdaHash": self.lambda_hash,
            "lambdaArn": self.lambda_arn, "evaluatorArn": self.evaluator_arn,
            "status": self.status, "error": self.error,
            "createdStamp": self.created_stamp,
            "deployed": deployed,
        }


def _registry() -> dict[str, Any]:
    from .store import get_store

    return get_store().read_root_json(_REGISTRY_FILE)


def _tenant_key() -> str:
    from .tenancy import current_tenant

    return current_tenant()


def list_reward_functions() -> list[dict[str, Any]]:
    """All reward functions for the current tenant (newest-first by insertion)."""
    by_tenant = _registry().get(_tenant_key(), {})
    return list(by_tenant.values())


def get_reward_function(reward_id: str) -> dict[str, Any] | None:
    return _registry().get(_tenant_key(), {}).get(reward_id)


def save_reward_function(rf: RewardFunction) -> None:
    from .store import get_store

    reg = _registry()
    reg.setdefault(_tenant_key(), {})[rf.id] = rf.to_dict()
    get_store().write_root_json(_REGISTRY_FILE, reg)


def delete_reward_function(reward_id: str) -> bool:
    from .store import get_store

    reg = _registry()
    tk = _tenant_key()
    if reward_id not in reg.get(tk, {}):
        return False
    del reg[tk][reward_id]
    get_store().write_root_json(_REGISTRY_FILE, reg)
    return True


def make_reward_function(name: str, *, snippet: str | None = None,
                         metric: str | None = None, stamp: str = "") -> RewardFunction:
    """Build a RewardFunction record from EITHER a user snippet OR a metric name
    (generates the snippet). Validates the snippet. Does NOT touch AWS — call
    deploy_reward_function to actually create the Lambda + Evaluator."""
    if bool(snippet) == bool(metric):
        raise RewardError("provide exactly one of `snippet` or `metric`")
    if metric:
        snippet = metric_reward_snippet(metric)
        kind = "metric"
    else:
        kind = "snippet"
    validate_snippet(snippet)
    nm = name.strip() or "reward"
    return RewardFunction(
        id=reward_id(nm, snippet, stamp), name=nm,
        snippet=snippet, kind=kind, metric=metric,
        lambda_hash=reward_hash(snippet), created_stamp=stamp,
    )


def make_reward_prompt(name: str, prompt: str, *, reward_model_id: str = "",
                       stamp: str = "") -> RewardFunction:
    """Build an RLAIF reward-PROMPT record (kind='reward_prompt'). Validates the
    prompt's placeholders. Unlike a verifiable reward there is NO Lambda/Evaluator
    to build — the prompt text is passed inline to RLAIFTrainer — so the record is
    immediately usable; deploy just flips status to 'deployed'."""
    validate_reward_prompt(prompt)
    jm = reward_model_id.strip()
    if jm and jm not in ALLOWED_JUDGE_MODELS:
        raise RewardError(
            f"unknown judge model {jm!r}; choose one of {list(ALLOWED_JUDGE_MODELS)} "
            "(or leave blank for the recipe default)"
        )
    nm = name.strip() or "reward-prompt"
    # Reuse reward_id (name + hash + stamp) keyed on the PROMPT text for a unique,
    # stable id; the `snippet` slot stays empty (no code).
    return RewardFunction(
        id=reward_id(nm, prompt, stamp), name=nm,
        snippet="", kind="reward_prompt", metric=None,
        prompt=prompt, reward_model_id=jm,
        lambda_hash=reward_hash(prompt),  # hash NAMES the S3 object + Evaluator (idempotent)
        created_stamp=stamp,
    )


def _set_status(reward_id: str, **fields: Any) -> None:
    """Patch a registry record's fields (status/error/arns) in place."""
    from .store import get_store

    reg = _registry()
    tk = _tenant_key()
    rec = reg.get(tk, {}).get(reward_id)
    if rec is None:
        return
    rec.update(fields)
    # A reward_prompt is usable once its REWARD_PROMPT Evaluator is registered;
    # a verifiable reward needs both its Lambda + Evaluator ARNs.
    if rec.get("kind") == "reward_prompt":
        rec["deployed"] = bool(rec.get("evaluatorArn"))
    else:
        rec["deployed"] = bool(rec.get("lambdaArn") and rec.get("evaluatorArn"))
    get_store().write_root_json(_REGISTRY_FILE, reg)


def run_deploy_reward_task(rid: str) -> None:
    """Worker task: build the zip, create the Lambda + Evaluator, and flip the
    registry record to deployed (or failed with the error). Runs off the request
    path (the V3-subprocess Evaluator.create + Lambda create are slow). `rid` is
    the record id; the Lambda/Evaluator are named by the snippet hash (lambdaHash)
    so identical snippets reuse one set of AWS resources."""
    from .obs import log_event

    rec = get_reward_function(rid)
    if rec is None:
        log_event("reward.deploy.not_found", level="WARNING", rewardId=rid)
        return
    # RLAIF reward prompts: NO Lambda, but the prompt MUST be registered as a
    # SageMaker Evaluator (type=REWARD_PROMPT) — the V3 RLAIFTrainer takes an
    # Evaluator ARN/name, not raw prompt text (a raw string fails the hubContent
    # name regex). Upload the prompt to S3 → register the Evaluator → store its ARN.
    if rec.get("kind") == "reward_prompt":
        _set_status(rid, status="deploying", error="")
        try:
            validate_reward_prompt(rec.get("prompt", ""))
            from .reward_deploy import deploy_reward_prompt

            prompt_key = rec.get("lambdaHash") or reward_hash(rec["prompt"])
            out = deploy_reward_prompt(rid, rec["prompt"], prompt_key=prompt_key)
            _set_status(rid, status="deployed", error="",
                        evaluatorArn=out["evaluatorArn"], promptS3Uri=out["promptS3Uri"])
            log_event("reward.deploy.done", rewardId=rid, kind="reward_prompt")
        except Exception as e:  # noqa: BLE001
            _set_status(rid, status="failed", error=str(e))
            log_event("reward.deploy.failed", level="WARNING", rewardId=rid, error=str(e))
        return
    _set_status(rid, status="deploying", error="")
    try:
        from .reward_deploy import deploy_reward_function

        zip_bytes = build_lambda_zip(rec["snippet"])
        lambda_key = rec.get("lambdaHash") or reward_hash(rec["snippet"])
        out = deploy_reward_function(rid, rec["name"], zip_bytes, lambda_key=lambda_key)
        _set_status(rid, status="deployed", error="",
                    lambdaArn=out["lambdaArn"], evaluatorArn=out["evaluatorArn"])
        log_event("reward.deploy.done", rewardId=rid)
    except Exception as e:  # noqa: BLE001 — record the failure for the UI
        _set_status(rid, status="failed", error=str(e))
        log_event("reward.deploy.failed", level="WARNING", rewardId=rid, error=str(e))
