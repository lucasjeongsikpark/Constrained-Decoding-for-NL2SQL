#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=a100

mkdir -p ./logs

# ==============================
# Parse Input Arguments
# ==============================
JOB_NAME=${1:-Qwen3_api}
MODEL_ID=${2:-Qwen/Qwen3.5-0.8B}
PORT=${3:-8082}

echo "🔹 Job Name: $JOB_NAME"
echo "🔹 Model ID: $MODEL_ID"
echo "🔹 Port: $PORT"

module purge
module load python/3.10

# ==============================
# Virtual Environment Setup
# ==============================
if [ ! -d ~/env_llama3 ]; then
    python -m venv ~/env_llama3
fi
source ~/env_llama3/bin/activate
pip install --upgrade pip
pip install torch transformers fastapi uvicorn accelerate sentencepiece pydantic outlines lmformatenforcer xgrammar

# ==============================
# Python App
# ==============================
python <<EOF
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

model_id = "${MODEL_ID}"
print(f"🔹 Loading {model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = 512
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    constraint: str | None = "none"  # none, outlines, lmfe, xgrammar
    regex_pattern: str | None = None
    grammar: str | None = None
    mode: str | None = "non-thinking"  # thinking or non-thinking


def build_terminators(tok):
    terminators = []
    if tok.eos_token_id is not None:
        terminators.append(tok.eos_token_id)
    for t in ["<|eot_id|>", "<|im_end|>", "<|endoftext|>"]:
        tok_id = tok.convert_tokens_to_ids(t)
        if tok_id is not None and tok_id != tok.unk_token_id:
            terminators.append(tok_id)
    if not terminators:
        terminators = [tok.pad_token_id]
    return list(dict.fromkeys([t for t in terminators if t is not None]))


def run_plain(prompt_ids, max_tokens, temperature, top_p):
    outputs = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        eos_token_id=build_terminators(tokenizer),
        do_sample=temperature is not None and temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    return outputs


def run_outlines(prompt_text: str, regex_pattern: str, max_tokens: int):
    try:
        import outlines
        from outlines.types import Regex
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=f"outlines not installed: {exc}")
    outlines_model = outlines.from_transformers(model, tokenizer)
    output = outlines_model(prompt_text, Regex(regex_pattern), max_new_tokens=max_tokens)
    text = output if isinstance(output, str) else str(output)
    return text


def run_lmfe(prompt_text: str, regex_pattern: str, max_tokens: int):
    try:
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        from lmformatenforcer.regex import create_regex_parser
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=f"lmformatenforcer not installed: {exc}")
    parser = create_regex_parser(regex_pattern)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=build_terminators(tokenizer),
        prefix_allowed_tokens_fn=prefix_fn,
    )
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def run_xgrammar(prompt_text: str, grammar: str, max_tokens: int, temperature: float | None, top_p: float | None):
    try:
        from xgrammar import LMGrammar, GenerationConfig
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=f"xgrammar not installed: {exc}")
    lm_grammar = LMGrammar.from_ebnf(tokenizer, grammar)
    cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=build_terminators(tokenizer),
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = lm_grammar.generate(model, inputs=inputs, generation_config=cfg)
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    start = time.perf_counter()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    enable_thinking = (req.mode or "non-thinking") == "thinking"
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model.device)

    max_tokens = req.max_tokens or 512
    temperature = req.temperature
    top_p = req.top_p
    constraint = (req.constraint or "none").lower()

    if constraint == "none":
        outputs = run_plain(prompt_ids, max_tokens, temperature, top_p)
        generated_ids = outputs[0][prompt_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    elif constraint == "outlines":
        if not req.regex_pattern:
            raise HTTPException(status_code=400, detail="regex_pattern required for outlines")
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        text = run_outlines(prompt_text, req.regex_pattern, max_tokens)
        generated_ids = tokenizer.encode(text, add_special_tokens=False)
    elif constraint == "lmfe":
        if not req.regex_pattern:
            raise HTTPException(status_code=400, detail="regex_pattern required for lmfe")
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        text = run_lmfe(prompt_text, req.regex_pattern, max_tokens)
        generated_ids = tokenizer.encode(text, add_special_tokens=False)
    elif constraint == "xgrammar":
        if not req.grammar:
            raise HTTPException(status_code=400, detail="grammar required for xgrammar")
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        text = run_xgrammar(prompt_text, req.grammar, max_tokens, temperature, top_p)
        generated_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported constraint: {constraint}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    completion_tokens = len(generated_ids)
    prompt_tokens = prompt_ids.shape[1]
    total_tokens = prompt_tokens + completion_tokens

    assistant_reply = text.strip()

    return JSONResponse({
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_reply}
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "latency_ms": round(float(latency_ms), 3),
        "constraint": constraint,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=${PORT})
EOF