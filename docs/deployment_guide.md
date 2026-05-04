# Medora Deployment Guide

---

## Overview

### Deployment Architecture

Medora runs on a single AWS EC2 g5.2xlarge instance serving the full stack:

```
Browser (any device)
    │
    │  HTTP (port 80)
    │
nginx (reverse proxy)
    ├── Static files (React frontend build)
    └── /api/* → uvicorn (FastAPI backend, port 8000)
                    ├── Agent pipeline (IntakeSession, TriageSession)
                    ├── RAG (ChromaDB + bi-encoder on CUDA + cross-encoder)
                    ├── PatientMemory + FeedbackStore
                    ├── Web search agent (SearXNG → LLM)
                    └── PostgreSQL (users, sessions, reports)

SearXNG (Docker, port 8080) — medical web search metaengine
```

### Instance Specifications

| Property | Value |
|---|---|
| Instance type | g5.2xlarge |
| GPU | NVIDIA A10G (24GB VRAM) |
| RAM | 32GB |
| vCPUs | 8 |
| EBS storage | 300GB |
| Region | eu-north-1 (Stockholm) |
| OS | Ubuntu 22.04 |
| Public IP | Elastic IP recommended (changes on stop/start without it) |

---

## Production Deployment Steps

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib nginx
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

Docker (for SearXNG) — already installed from benchmarking phase.

### 2. PostgreSQL Setup

```bash
sudo -u postgres createuser ubuntu
sudo -u postgres createdb medora -O ubuntu
```

Configure trust authentication for local connections:
```bash
sudo nano /etc/postgresql/14/main/pg_hba.conf
```
Change `peer` → `trust` for local connections and `scram-sha-256` → `trust` for 127.0.0.1.

```bash
sudo systemctl restart postgresql
```

### 3. SearXNG (Docker)

```bash
mkdir -p ~/searxng && cd ~/searxng
```

Create `docker-compose.yml`:
```yaml
version: "3"
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8080:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml
    restart: unless-stopped
```

Create `settings.yml`:
```yaml
use_default_settings: true
server:
  secret_key: medora-searxng-secret-key-2026
  bind_address: "0.0.0.0"
  port: 8080
search:
  formats:
    - html
    - json
```

```bash
docker compose up -d
curl -s "http://localhost:8080/search?q=test&format=json" | head -c 100
```

### 4. Application Code

```bash
cd ~/Medora
git pull origin main
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install email-validator
```

### 5. Backend Environment

Create `backend/.env`:
```
DATABASE_URL=postgresql://ubuntu@localhost/medora
SECRET_KEY=<generate-with-python3 -c "import secrets; print(secrets.token_hex(32))">
OPENAI_API_KEY=<your-key>
SEARXNG_BASE_URL=http://localhost:8080
```

### 6. Database Migrations

```bash
cd ~/Medora/backend
alembic upgrade head
```

### 7. Bootstrap Users

```bash
ADMIN_EMAIL=admin@medora.com ADMIN_PASSWORD=admin123 python create_admin.py
python create_doctor.py  # creates doctor@medora.com / doctor123
```

### 8. Data Files (not in git)

These files are in `.gitignore` and must be SCP'd from the development machine:

```bash
# From local Mac:
scp -i ~/.ssh/medora-key.pem \
  data/structured_symptoms/tmt_symptoms_gpt4o.json \
  ubuntu@<ip>:~/Medora/data/structured_symptoms/

scp -i ~/.ssh/medora-key.pem \
  data/chunks/tmt_chunks_structured.json \
  ubuntu@<ip>:~/Medora/data/chunks/

scp -i ~/.ssh/medora-key.pem \
  data/embeddings/tmt_chunk_embeddings.npz \
  ubuntu@<ip>:~/Medora/data/embeddings/

scp -i ~/.ssh/medora-key.pem \
  data/embeddings/embedding_metadata.json \
  ubuntu@<ip>:~/Medora/data/embeddings/
```

ChromaDB must be rebuilt on EC2 (not portable across machines):
```bash
python embeddings/build_vector_store.py --force
```

### 9. Build Frontend

```bash
cd ~/Medora/frontend
npm install
npm run build
```

### 10. nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/medora
```

```nginx
server {
    listen 80;
    server_name _;

    root /home/ubuntu/Medora/frontend/dist;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

```bash
sudo ln -sf /etc/nginx/sites-available/medora /etc/nginx/sites-enabled/medora
sudo rm -f /etc/nginx/sites-enabled/default
chmod 755 /home/ubuntu
chmod -R 755 /home/ubuntu/Medora/frontend/dist
sudo nginx -t
sudo systemctl restart nginx
```

### 11. Start Backend

```bash
cd ~/Medora/backend
source ~/.venv/bin/activate
export SEARXNG_BASE_URL=http://localhost:8080
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > ~/backend.log 2>&1 &
```

Monitor startup (wait for "Application startup complete"):
```bash
tail -f ~/backend.log
```

### 12. AWS Security Group

Required inbound rules:

| Type | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | 0.0.0.0/0 | SSH access |
| HTTP | 80 | 0.0.0.0/0 | Web app |

Port 8080 (SearXNG) should NOT be publicly exposed — it's accessed only by the backend on localhost.

---

## Accessing the App

```
http://<ec2-public-ip>
```

### Default Accounts

| Role | Email | Password |
|---|---|---|
| Admin | admin@medora.com | admin123 |
| Doctor | doctor@medora.com | doctor123 |
| Patient | (sign up in browser) | — |

---

## GPU Configuration

The bi-encoder (`embeddinggemma-300m-medical`) runs on CUDA (NVIDIA A10G). The cross-encoder (`bge-reranker-v2-m3`) falls back to CPU if CUDA encounters issues with the model.

Device detection priority: CUDA → MPS → CPU (handled by `detect_device()` in `rag/reranker.py`).

---

## Operations

### Restart Backend

```bash
kill $(pgrep -f "uvicorn app.main")
cd ~/Medora/backend
source ~/.venv/bin/activate
export SEARXNG_BASE_URL=http://localhost:8080
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > ~/backend.log 2>&1 &
```

### View Logs

```bash
tail -f ~/backend.log
```

### Update Code

```bash
cd ~/Medora
git pull origin main
# Restart backend (see above)
# If frontend changed: cd frontend && npm run build
```

### Reset Database

```bash
kill $(pgrep -f "uvicorn app.main")
sudo -u postgres dropdb medora
sudo -u postgres createdb medora -O ubuntu
cd ~/Medora/backend
alembic upgrade head
ADMIN_EMAIL=admin@medora.com ADMIN_PASSWORD=admin123 python create_admin.py
python create_doctor.py
# Restart backend
```

### Restart SearXNG

```bash
cd ~/searxng
docker compose restart
```

---

## SSH Access

```bash
ssh -i ~/.ssh/medora-key.pem ubuntu@<check-current-ip>
```

The EC2 public IP changes on every stop/start unless an Elastic IP is assigned.

---

## Cost Management

| Resource | Cost | Notes |
|---|---|---|
| g5.2xlarge (running) | ~$1.29/hour | Billed only while instance is running |
| EBS 300GB | ~$24/month | Billed continuously, even when instance is stopped |
| OpenAI API | ~$0.01-0.05/session | GPT-4o-mini for agents + web search |

**Always stop the instance when not in use:**
AWS Console → EC2 → Instances → select instance → Instance state → Stop

Stopping preserves all data on disk (PostgreSQL, ChromaDB, SearXNG Docker image, application code). Starting resumes from the same state — but the backend and SearXNG must be restarted manually.

### After Instance Restart

```bash
# 1. Start SearXNG
cd ~/searxng && docker compose up -d

# 2. Start backend
cd ~/Medora/backend
source ~/.venv/bin/activate
export SEARXNG_BASE_URL=http://localhost:8080
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > ~/backend.log 2>&1 &

# 3. nginx starts automatically
```
