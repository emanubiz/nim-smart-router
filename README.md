# nim-smart-router

**Lightweight LLM proxy** that routes requests across NVIDIA NIM models with automatic fallback on rate limits (429) and errors.

OpenAI-compatible → works with Claude Code, Cursor, Windsurf, Pi, any coding agent.

## Quickstart

**Linux / macOS**

```bash
cp .env.example .env          # paste your NVIDIA_NIM_API_KEY
pip install -r requirements.txt
chmod +x start.sh && ./start.sh   # → http://127.0.0.1:4000
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

| # | Name | Tier | SWE-Bench | GPQA | LiveCodeBench |
|---|------|------|-----------|------|---------------|
| 1 | minimax-m3 | 🥇 TIER 1 | 59.0% Pro | 92.7% | — |
| 2 | deepseek-v4-pro | 🥇 TIER 1 | 80.6% Verif. | — | 93.5 |
| 3 | kimi-k2.6 | 🥇 TIER 1 | 80.2% Verif. | 90.5% | — |
| 4 | nemotron-3-ultra-550b | 🥇 TIER 1 | ~67% Verif. | 87.0% | 89.0% |
| 5 | qwen3.5-397b | 🥈 TIER 2 | 76.4% Verif. | 88.4% | 83.6% |
| 6 | nemotron-3-super-120b | 🥈 TIER 2 | ~60% Verif. | 82.7% | 81.2% |
| 7 | qwen3-coder-480b | 🥈 TIER 2 | — | — | — |
| 8 | qwen3-235b | 🥈 TIER 2 | — | — | — |
| 9 | deepseek-v4-flash | 🥉 TIER 3 | — | — | — |
| 10 | glm-5.1 | 🥉 TIER 3 | — | — | — |
| 11 | step-3.7-flash | 🥉 TIER 3 | 56.3% Pro | — | — |
| 12 | step-3.5-flash | 🥉 TIER 3 | 51.3% Pro | — | — |
| 13 | llama-nemotron-super-49b | 🔄 BACKUP | — | — | — |
| 14 | llama-3.1-70b | 🔄 BACKUP | — | — | — |

> **Why no Mistral Large 3 675B?** Its GPQA Diamond is only **43.9%** —
> worse than most 7B models. Not worth the slot.

Use `model="auto"`, `model="best-available"`, or `model="best"` for automatic
selection. With **sticky routing**, if a top-4 model (minimax-m3, deepseek-v4-pro,
kimi-k2.6, nemotron-3-ultra-550b) responded successfully last time, the next
"auto" request starts from that model instead of always falling back to #1.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + tools) |
| `POST` | `/chat/completions` | Same as above, alias without `/v1` |
| `GET`  | `/v1/models` | List models with their priority |
| `GET`  | `/health`, `/v1/health` | Health check + fallback chain |

## Fallback

Calling `model="minimax-m3"` but it returns 429? The router transparently tries
deepseek-v4-pro → kimi-k2.6 → nemotron-3-ultra-550b → qwen3.5-397b →
nemotron-3-super-120b → qwen3-coder-480b → qwen3-235b → deepseek-v4-flash →
glm-5.1 → step-3.7-flash → step-3.5-flash → nemotron-super-49b →
llama-3.1-70b.

Your agent gets one response, no errors. Models in cooldown (30s after a 429)
are skipped automatically.

## Features

- **model: "auto"** — picks the best available model
- **Sticky routing** — remembers the last successful top-tier model, reuses it on next "auto" request
- **Tool/function calling** — full passthrough
- **Streaming** — real SSE passthrough
- **OpenAI format** — response identical to direct call
- **Fallback** — automatic on 429 / timeout / error (up to 10 fallbacks, 14 models total)
- **Cooldown** — failing models rest 30s before retry
- **Per-call timeout** — 60s ceiling per upstream call, so a stuck endpoint triggers fallback instead of hanging
- **Graceful stream errors** — if every model fails mid-stream, the error is delivered as a valid SSE event (never a broken stream)
- **Graceful shutdown** — closes cleanly on Ctrl+C, terminal close, or kill signal

## Requirements

- Python 3.11+
- `pip install litellm fastapi uvicorn`

## Project

```
nim-smart-router/
├── server.py           # one file, zero bloat
├── start.sh            # Linux/macOS launcher (loads .env, runs server)
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
