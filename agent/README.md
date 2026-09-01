# Dataset investigator agent

The LLM-reasoning component of the platform, packaged as an [Amazon Bedrock
AgentCore](https://aws.amazon.com/bedrock/agentcore/) Runtime. AgentCore POSTs a
JSON payload to the `@app.entrypoint` function in `agent.py` and returns the dict
it produces.

It is stateless. The backend computes the deterministic dataset profile
(`backend/app/profiler.py`) and passes it in the payload, so the runtime needs no
data access of its own — only Bedrock model invocation. Reasoning runs through
[Strands](https://strandsagents.com/) `Agent` + `BedrockModel` at temperature 0.

## Actions

`payload["action"]` selects the behavior:

| Action | Required payload | Returns |
| --- | --- | --- |
| `questions` | `profile` | `{questions: [...], summary}` — 3–6 follow-up questions about the dataset, each tagged with the data-quality facet that motivated it |
| `proposal` | `profile`, `answers` | `{taskType, rankMetric, alsoWatch, cutoffGuidance, flaggedIssues, rationale, appliedAnswers}` — a config proposal that pre-fills the fine-tune form and locks the eval metric |
| `triage` | `context` (`model`, `failureReason`, `logTail`, `config`, `classification`) | Plain-language diagnosis of a failed job plus one recommended fix |
| `interpret` | `leaderboard` (`rows`, `baselines`), optional `priorities` | Which model to ship, given the run's results |
| `reward_author` | `goal`, `profile`, optional `priorResult` | `{draftPrompt, rewardModelId, samples, scoreSpread, rationale, iterations, judgeCalls, warnings}` |

An unknown action returns `{"error": ...}` rather than raising.

`questions` and `proposal` implement the "ask only what you can't derive" rule
documented at the top of `core.py`: the agent may ask about task- and
human-facet dimensions, and never about anything the profiler already computes.

`reward_author` is the only multi-turn action. It drafts an RLAIF judge rubric,
scores candidate completions with a real judge model through the `set_rubric` /
`score_candidate` tools, and iterates under a hard cap on judge calls. The score
spread it reports is recomputed server-side from the captured scores rather than
taken from the model's reply.

## Model selection

Default reasoning model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (the
cross-region inference profile). A payload may carry `modelId` to override it for
that invocation only — the entrypoint calls `set_reasoning_model()` before
dispatch and the value resets each request, so one caller's choice cannot leak
into the next.

The judge used by `reward_author` is separate and restricted to the open-weight
models in `judge_tools.ALLOWED_JUDGE_MODELS`.

## Layout

```
agent.py                              AgentCore entrypoint + action dispatch
src/dataset_investigator/core.py      prompts, facet gate, the five actions
src/dataset_investigator/judge_tools.py  judge scoring helpers
src/dataset_investigator/aws_user_agent.py  solution user-agent suffix
tests/                                pytest suite (Bedrock mocked)
```

`judge_tools.py` is a deliberate copy of the same-named helpers in
`backend/app/reward_functions.py`, because the runtime container packages only
this directory and cannot import the backend `app` package. Edits must be
mirrored in both files; `backend/tests/test_reward_author_parity.py` fails if
they drift.

## Environment variables

- `AWS_REGION` — region for Bedrock calls. AgentCore sets it in the runtime;
  defaults to `us-east-1`.
- `USER_AGENT_STRING` — optional user-agent suffix appended to this process's AWS
  calls for usage attribution. Unset means unattributed, never an error. Because
  this runtime deploys outside the CDK app, nothing sets it automatically; pass it
  at launch time if you want the attribution.

## Local development

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv run agent.py                  # serves the same contract on :8080
curl -X POST localhost:8080/invocations \
  -H 'content-type: application/json' \
  -d '{"action": "questions", "profile": {...}}'

uv run -m pytest tests/ -q       # no AWS calls; Bedrock is mocked
```

## Deployment

```bash
agentcore configure --entrypoint agent.py
agentcore launch
```

The runtime is deployed by the `agentcore` CLI, not by the CDK app in `infra/`.
Its id gets a random suffix, so the backend resolves the ARN by the stable
runtime name (`dataset_investigator`) through the AgentCore control plane and
caches it. Override with `SLM_AGENT_RUNTIME_NAME` or, for a full ARN,
`SLM_AGENT_RUNTIME_ARN`. See `backend/app/investigator.py`.

The runtime's execution role needs permission to invoke the Bedrock models above
— both the reasoning model and any judge model `reward_author` is asked to use.
