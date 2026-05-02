# Medora Deployment Guide

**Living document** — updated incrementally as deployment work progresses.

---

## Overview

### Deployment Architecture

Medora runs across two environments:

| Environment | Machine | Purpose |
|---|---|---|
| Development | Mac M1 Pro | API model development, benchmarking with OpenAI models, RAG pipeline iteration |
| Production (in progress) | AWS EC2 g5.2xlarge | Open-source model hosting via Ollama, GPU benchmarking |

### Two-Tier Model Strategy

- **Mac (local):** API models (GPT-5.4-mini, GPT-5.4, GPT-4o) via OpenAI. No GPU required. Requires `OPENAI_API_KEY`.
- **EC2 (cloud):** Open-source models (Llama, Gemma, Phi, MedLlama) via Ollama. A10G GPU. Zero per-token cost.

---

## EC2 Instance Setup

### Instance Specifications

| Property | Value |
|---|---|
| Instance type | g5.2xlarge |
| GPU | NVIDIA A10G (24GB VRAM) |
| RAM | 32GB |
| EBS storage | 300GB |
| Region | eu-north-1 (Stockholm) |
| OS | Ubuntu 22.04 |

### GPU Quota

AWS G instance type quota was increased to 8 vCPUs for G and VT instances in eu-north-1 via a support request. The default quota is 0.

### Security Group

SSH (port 22) open. No additional ports exposed yet.

### Steps Performed

1. Changed instance type from t3.micro to g5.2xlarge in the AWS console.
2. Expanded EBS volume from 50GB to 300GB (required for Ollama models — 70B model alone is 42GB).
3. Installed NVIDIA drivers:
   ```bash
   sudo apt install nvidia-driver-570 nvidia-utils-570
   ```
4. Rebooted to load the drivers.
5. Verified GPU with `nvidia-smi` — confirms A10G, CUDA 13.0.

---

## Docker

Docker v29.4.2 is installed on EC2. Containerization of the Medora pipeline is pending — the current setup runs directly on the host.

---

## Ollama Setup

### Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Models Downloaded

| Model | Model ID | Disk size |
|---|---|---|
| Llama 3.1 70B (4-bit quantized) | `llama3.1:70b-instruct-q4_K_M` | 42GB |
| Gemma 2 27B | `gemma2:27b` | 15GB |
| Phi-4 14B | `phi4:14b` | 9GB |
| Llama 3.1 8B | `llama3.1:8b` | 5GB |
| MedLlama2 7B | `medllama2:7b` | 4GB |

```bash
ollama pull llama3.1:70b-instruct-q4_K_M
ollama pull gemma2:27b
ollama pull phi4:14b
ollama pull llama3.1:8b
ollama pull medllama2:7b
```

### Verification

Verified that `llama3.1:70b-instruct-q4_K_M` runs correctly by sending a medical question and confirming a clinically correct response.

---

## Medora Codebase Deployment

### Steps Performed

1. Cloned the repository:
   ```bash
   git clone https://github.com/KarimHabbal1/Medora.git
   ```

2. Checked out the benchmarking branch:
   ```bash
   git checkout benchmarking/model-comparison
   ```

3. Set up Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Installed dependencies:
   ```bash
   pip install -r requirements.txt
   pip install datasets langchain-community
   ```

5. SCP'd data files from Mac (see Data Transfer section below).

6. Rebuilt the ChromaDB vector store on EC2:
   ```bash
   python embeddings/build_vector_store.py --force
   ```

7. Created `.env` with the OpenAI API key (used by the judge model during benchmarking):
   ```
   OPENAI_API_KEY=sk-...
   ```

---

## LLM Configuration

### The `make_llm()` Factory

Medora uses a unified `make_llm()` factory function that auto-detects whether to use OpenAI or Ollama based on the model name:

- Model names containing `":"` (e.g., `gemma2:27b`, `llama3.1:8b`) → Ollama
- Model names without `":"` (e.g., `gpt-5.4-mini`, `gpt-5.4`) → OpenAI

### CLI Flags

| Flag | Description |
|---|---|
| `--model` | Model name (e.g., `gpt-5.4-mini` or `gemma2:27b`) |
| `--provider` | Explicit provider override (`openai` or `ollama`) |
| `--ollama-url` | Ollama server base URL (default: `http://localhost:11434`) |

### Usage Examples

```bash
# Local development with OpenAI
python agents/intake_agent.py --model gpt-5.4-mini

# EC2 with Ollama (run on EC2 directly)
python agents/intake_agent.py --model gemma2:27b
python agents/triage_agent.py --model llama3.1:8b --query "chest pain"

# Remote Ollama from Mac (substitute actual EC2 IP)
python agents/intake_agent.py --model gemma2:27b --ollama-url http://<ec2-ip>:11434
```

---

## Running Benchmarks

### On Mac (API Models)

```bash
# Full pipeline benchmark — all test sets, GPT-5.4-mini + GPT-5.4
python evaluation/pipeline_benchmark.py --run --profile api

# Raw LLM benchmark (MedCaseReasoning test set)
python evaluation/benchmark.py --profile api

# Ceiling benchmark — GPT-5.4 only, 10 cases per set
python evaluation/benchmark.py --profile api-ceiling
```

### On EC2 (Ollama Models)

```bash
# Activate venv first
source .venv/bin/activate

# Full pipeline benchmark — all Ollama models, all test sets
python evaluation/pipeline_benchmark.py --run --profile ollama

# Raw LLM benchmark
python evaluation/benchmark.py --profile ollama
```

### Targeting EC2 from Mac (Remote Ollama)

```bash
python evaluation/pipeline_benchmark.py --run --profile ollama \
  --ollama-url http://<ec2-ip>:11434

python evaluation/benchmark.py --profile ollama \
  --ollama-url http://<ec2-ip>:11434
```

---

## SSH Access

```bash
ssh -i ~/.ssh/medora-key.pem ubuntu@<check-current-ip>
```

**Important:** The EC2 public IP changes on every stop/start. Check the AWS console (EC2 → Instances → select instance → Public IPv4 address) for the current IP before connecting.

---

## Cost Management

| Resource | Cost | Notes |
|---|---|---|
| g5.2xlarge (running) | ~$1.29/hour | Billed only while instance is running |
| EBS 300GB | ~$24/month | Billed continuously, even when instance is stopped |
| Ollama models (disk) | Included in EBS cost | No re-download needed after stop/start |

**Always stop the instance when not in use:**
AWS Console → EC2 → Instances → select instance → Instance state → Stop

Stopping preserves all data on disk (Ollama models, cloned repo, venv). Starting resumes from the same state — no re-setup required.

---

## Data Transfer (SCP)

The following files are in `.gitignore` and must be manually transferred to EC2 after a fresh clone. They are too large for git (embeddings file) or contain generated data that is not committed.

### Mac to EC2

```bash
# Substitute the actual EC2 IP

# Textbook chunks
scp -i ~/.ssh/medora-key.pem \
  data/chunks/tmt_chunks_structured.json \
  ubuntu@<ip>:~/Medora/data/chunks/

# Evaluation test cases
scp -i ~/.ssh/medora-key.pem \
  data/evaluation/textbook_test_cases.json \
  ubuntu@<ip>:~/Medora/data/evaluation/

scp -i ~/.ssh/medora-key.pem \
  data/evaluation/medqa_filtered_cases.json \
  ubuntu@<ip>:~/Medora/data/evaluation/

# Embeddings
scp -i ~/.ssh/medora-key.pem \
  data/embeddings/tmt_chunk_embeddings.npz \
  ubuntu@<ip>:~/Medora/data/embeddings/

scp -i ~/.ssh/medora-key.pem \
  data/embeddings/embedding_metadata.json \
  ubuntu@<ip>:~/Medora/data/embeddings/
```

### Rebuild ChromaDB on EC2

After SCP'ing the chunks and embeddings, rebuild the vector store:

```bash
python embeddings/build_vector_store.py --force
```

This is required because ChromaDB stores are not portable across machines — the index must be rebuilt from the chunks and embeddings on each new host.

---

## Remaining Deployment Tasks

| Task | Status | Notes |
|---|---|---|
| Dockerize the full pipeline | Pending | Docker is installed, containerization not started |
| systemd service or Docker Compose | Pending | Ollama currently started manually |
| Ollama on boot (systemd) | Pending | Needs `ollama serve` as a systemd unit |
| Elastic IP | Pending | Prevents IP change on restart |
| HTTPS / external access | Pending | Not needed until external deployment |
| Benchmark EC2 runs (all Ollama models) | Pending | EC2 is ready; benchmarks not yet run |
| GPT-5.4 ceiling benchmark | Pending | Mac; cost-gated at 10 cases per set |
