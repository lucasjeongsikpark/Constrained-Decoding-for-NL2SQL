#!/bin/bash
# Simple smoke test against the remote Qwen FastAPI server.
# Usage: BASE_URL=http://localhost:8082/v1 MODEL_NAME=Qwen/Qwen3.5-0.8B ./call_qwen.sh "your prompt"

BASE_URL=${BASE_URL:-"http://localhost:8082/v1"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3.5-0.8B"}
PROMPT=${1:-"Summarize LangGraph in one sentence."}

python - <<'PY' "$BASE_URL" "$MODEL_NAME" "$PROMPT"
import sys
import os
from openai import OpenAI

base_url, model_name, prompt = sys.argv[1:4]

client = OpenAI(base_url=base_url, api_key="EMPTY")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

resp = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

print("--- Request ---")
print(f"BASE_URL: {base_url}")
print(f"MODEL:    {model_name}")
print(f"PROMPT:   {prompt}")

print("\n--- Response ---")
choice = resp.choices[0].message.content
print(choice.strip())

usage = getattr(resp, "usage", None)
if usage:
    print("\n--- Usage ---")
    print(f"prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}, total_tokens={usage.total_tokens}")

latency_ms = getattr(resp, "latency_ms", None)
if latency_ms is not None:
    print(f"latency_ms={latency_ms}")
PY