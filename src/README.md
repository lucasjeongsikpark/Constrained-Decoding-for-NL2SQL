# 🚀 CARC LLMs API Deployment Guide

This guide explains how to **deploy and access LLM APIs on the CARC GPU cluster** (e.g., Mistral & Qwen)  
using SLURM + FastAPI + OpenAI-compatible endpoints.

---

## 🖥️ [Server Setup on CARC]

### 1️⃣ Configure Hugging Face Access

```bash
export HF_TOKEN={your_huggingface_token}
```

### 2️⃣ Prepare Deployment Scripts

Copy & paste the content into each file:

```bash
vi deploy_qwen.sh
vi deploy_mistral.sh
```

### 3️⃣ Launch Models via SLURM

```bash
sbatch deploy_qwen.sh
sbatch deploy_mistral.sh
```

> ⏳ **Note:**  
> The first time you load each model, it may take **up to 30 minutes**.  
> You can monitor the startup and inference logs in the `logs/` directory.

---

## 💻 [Client Access]

### 1️⃣ Connect via SSH Tunnel

Run the following command on your **local machine** to create a secure tunnel  
to your CARC node running the API server.

You can check your **NODE_NAME (Node List)** here:  
👉 [CARC Active Jobs Dashboard](https://ondemand.carc.usc.edu/pun/sys/dashboard/activejobs)

```bash
./connect_carc_api.sh {NODE_NAME} {PORT_NUMBER} {USC_EMAIL_PREFIX}
```

#### 🔹 Example

```bash
./connect_carc_api.sh a04-20 8082 jeongsik   # for Qwen
./connect_carc_api.sh a04-19 8083 jeongsik   # for Mistral
```

---

### 2️⃣ Run Inference from Local Terminal

Once the SSH tunnel is active, open another terminal and run:

```bash
python call_mistral.py
python call_qwen.py
```

Both scripts send OpenAI-style API requests to:

- `http://localhost:8082/v1/chat/completions` → Qwen3-8B
- `http://localhost:8083/v1/chat/completions` → Mistral-7B-Instruct

---

## ✅ Quick Health Check (Optional)

To verify that your server is running correctly:

```bash
curl http://localhost:8082/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize LangGraph in one sentence."}
    ]
  }'
```

```bash
curl http://localhost:8083/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize LangGraph in one sentence."}
    ]
  }'
```

If you see a JSON response with `"role": "assistant"`, your endpoint is working perfectly 🎯

---
