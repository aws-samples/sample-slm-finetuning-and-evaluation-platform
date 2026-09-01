# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Self-contained SageMaker inference server for an exported SLM fine-tune.

Runs on the SAME LLaMA-Factory image the model was trained on (see Dockerfile),
so transformers/torch versions match the weights exactly. Generation mirrors the
platform's eval harness (container/eval.py): apply the model's chat template, then
model.generate.

SageMaker's "bring your own container" contract is just two HTTP routes:
  GET  /ping         → 200 when the model is loaded (health check)
  POST /invocations  → run inference on the request body
We serve them with FastAPI + uvicorn (both ship in the LLaMA-Factory base image),
which is far simpler and more robust than the sagemaker-inference/MMS toolkit.

Two delivery modes:
  * merged  — SLM_MODEL_DIR holds the standalone merged model; loaded directly.
  * adapter — SLM_MODEL_DIR holds a LoRA adapter; the base is read from
              SLM_BASE_MODEL and applied with PEFT (needs HF_TOKEN for gated bases).

Request JSON:  {"prompt": "...", "max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0}
Response JSON: {"generated_text": "..."}
"""

import json
import os

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer

# SageMaker mounts the model.tar.gz contents here.
MODEL_DIR = os.environ.get("SLM_MODEL_DIR", "/opt/ml/model")
BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "").strip()  # set only in adapter mode
# Whether to execute modeling code shipped inside the model repo (`auto_map`).
# It runs unsandboxed in this endpoint's container, so it is OFF unless the
# exporter set it for a base whose architecture requires it (see manifest.json).
TRUST_REMOTE_CODE = os.environ.get("SLM_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes", "on")

app = FastAPI()
_STATE: dict = {}  # lazily-loaded {model, tokenizer}


def _load() -> dict:
    """Load tokenizer + model once (cached). bf16 on GPU, fp32 on CPU."""
    if _STATE:
        return _STATE
    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if BASE_MODEL:
        # Adapter mode: base from HF (HF_TOKEN env covers gated), then the adapter.
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=dtype, device_map=device_map, trust_remote_code=TRUST_REMOTE_CODE
        )
        model = PeftModel.from_pretrained(base, MODEL_DIR)
    else:
        # Merged mode: MODEL_DIR is a fully standalone model.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=TRUST_REMOTE_CODE)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=dtype, device_map=device_map, trust_remote_code=TRUST_REMOTE_CODE
        )
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _STATE["model"], _STATE["tokenizer"] = model, tokenizer
    return _STATE


@app.get("/ping")
def ping() -> Response:
    """SageMaker health check — 200 only once the model is loaded."""
    try:
        _load()
        return Response(status_code=200)
    except Exception:  # noqa: BLE001 — not ready yet
        return Response(status_code=503)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    state = _load()
    model, tokenizer = state["model"], state["tokenizer"]
    body = await request.body()
    data = json.loads(body or b"{}")
    prompt = data.get("prompt", "")
    max_new_tokens = int(data.get("max_new_tokens", 256))
    temperature = float(data.get("temperature", 0.0))
    top_p = float(data.get("top_p", 1.0))

    # Same formatting as training/eval: apply the model's chat template; fall back
    # to raw encoding for models without one (matches eval.py's behaviour).
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "top_p": top_p}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    result = {"generated_text": tokenizer.decode(gen, skip_special_tokens=True).strip()}
    return Response(content=json.dumps(result), media_type="application/json")


if __name__ == "__main__":
    # SageMaker invokes the image as `serve`; we listen on the required port 8080.
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
