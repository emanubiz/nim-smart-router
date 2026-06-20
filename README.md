# nim-smart-router

**Lightweight LLM proxy** that routes requests across NVIDIA NIM models with automatic fallback on rate limits (429) and errors.

OpenAI-compatible → works with Claude Code, Cursor, Windsurf, Pi, any coding agent.

## Quickstart

**Linux / macOS**

```bash
cp .env.example .env          # paste your NVIDIA_NIM_API_KEY
pip install -r requirements.txt
python server.py               # → http://127.0.0.1:4000
```

**Windows**

```bat
copy .env.example .env         :: paste your NVIDIA_NIM_API_KEY
pip install -r requirements.txt
start.bat                      :: loads .env and launches the server
```

Then point any OpenAI-compatible client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="any")
# model="auto" → picks the best model, falls back on 429
r = client.chat.completions.create(model="auto", messages=[{"role":"user","content":"Hi!"}])
```

> The proxy authenticates to NVIDIA NIM with `NVIDIA_NIM_API_KEY` from `.env`.
> The `api_key` you pass from the client is ignored — use any non-empty string.

## Models (in order of priority)

| # | Name | Tier |
|---|------|------|
| 1 | kimi-k2.6 | 🥇 TOP |
| 2 | deepseek-v4-pro | 🥇 TOP |
| 3 | deepseek-v4-flash | 🥈 MID |
| 4 | minimax-m3 | 🥈 MID |
| 5 | glm-5.1 | 🥈 MID |
| 6 | step-3.5-flash | 🥉 LOW |
| 7 | step-3.7-flash | 🥉 LOW |
| 8 | llama-nemotron-super-49b | 🔄 BACKUP |
| 9 | llama-3.1-70b | 🔄 BACKUP |

Use `model="auto"`, `model="best-available"`, or `model="best"` for automatic
selection — all three map to the top-priority model. You can also request any
model by name (e.g. `model="glm-5.1"`); fallback still applies from that point
down the chain.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + tools) |
| `POST` | `/chat/completions` | Same as above, alias without `/v1` |
| `GET`  | `/v1/models` | List models with their priority |
| `GET`  | `/health`, `/v1/health` | Health check + fallback chain |

## Fallback

Calling `model="kimi-k2.6"` but it returns 429? The router transparently tries deepseek-v4-pro → deepseek-v4-flash → minimax-m3 → glm-5.1 → step-3.5-flash → step-3.7-flash → nemotron → llama-70b. Your agent gets one response, no errors.

## Features

- **model: "auto"** — picks the best available model
- **Tool/function calling** — full passthrough
- **Streaming** — real SSE passthrough
- **OpenAI format** — response identical to direct call
- **Fallback** — automatic on 429 / timeout / error
- **Cooldown** — failing models rest 30s before retry
- **Per-call timeout** — 60s ceiling per upstream call, so a stuck endpoint triggers fallback instead of hanging
- **Graceful stream errors** — if every model fails mid-stream, the error is delivered as a valid SSE event (never a broken stream)

## Requirements

- Python 3.11+
- `pip install litellm fastapi uvicorn`

## Project

```
nim-smart-router/
├── server.py           # one file, zero bloat
├── start.bat           # Windows launcher (loads .env, runs server)
├── requirements.txt    # 3 dependencies
├── .env.example        # API key template
├── .env                # your key (gitignored)
├── .gitignore
├── LICENSE             # MIT
├── RATE_LIMITING.md    # design note (rate-limit strategy, not yet wired in)
└── README.md
```

Works on any OS. Fits on a Raspberry Pi.

Based on [LiteLLM](https://github.com/BerriAI/litellm) — the 50k★ LLM gateway.  
Powered by [NVIDIA NIM](https://build.nvidia.com) — free inference endpoints.
