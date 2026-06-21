"""
nim-smart-router
~~~~~~~~~~~~~~~~
Transparent LLM routing across NVIDIA NIM models with automatic fallback
on rate-limit (429) or errors. OpenAI-compatible — plug into any coding agent.

Usage:
    cp .env.example .env   # add your NVIDIA_NIM_API_KEY
    pip install -r requirements.txt
    python server.py
"""

import os
import sys
import json
import time
import asyncio
import atexit
import signal
import logging
import contextlib
from typing import Optional

# Ensure we use pip-installed litellm, not a local clone
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]

# Skip cost-map download on every startup (slows init)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from litellm import Router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nim-router")

router: Optional[Router] = None

# Sticky routing: remember which model last succeeded.
# If it's among the top N, use it as the starting point for the next "auto"
# request instead of always falling back to MODEL_PRIORITY[0].
_last_successful_model: Optional[str] = None

# === MODEL CONFIG ===

# Priority-ordered list: trial order on fallback
# Ranking based on public benchmarks (SWE-Bench Pro/Verified, LiveCodeBench, GPQA — June 2026)
MODEL_PRIORITY = [
    "minimax-m3",               # TIER 1 - Frontier: SWE-Bench Pro 59.0% (best open), GPQA 92.7%
    "deepseek-v4-pro",          # TIER 1 - Frontier: SWE-Bench Verified 80.6%, LiveCodeBench 93.5
    "kimi-k2.6",                # TIER 1 - Frontier: SWE-Bench Verified 80.2%, GPQA 90.5%, strong agentic
    "nemotron-3-ultra-550b",    # TIER 1 - Frontier: LiveCodeBench 89.0%, GPQA 87.0%, 550B MoE
    "qwen3.5-397b",             # TIER 2 - Strong: SWE-Bench Verified 76.4%, GPQA 88.4%, 397B MoE
    "nemotron-3-super-120b",    # TIER 2 - Strong: SWE-Bench Verified ~60%, LiveCodeBench 81.2%
    "qwen3-coder-480b",         # TIER 2 - Coder: 480B MoE specialist, Claude Sonnet-level coding
    "qwen3-235b",               # TIER 2 - Strong: 235B MoE, top reasoning & multilingual
    "deepseek-v4-flash",        # TIER 3 - Fast V4 variant, still frontier quality
    "glm-5.1",                  # TIER 3 - Mid: Zhipu AI
    "step-3.7-flash",           # TIER 3 - SWE-Bench Pro 56.3%, ClawEval-1.1 #1
    "step-3.5-flash",           # TIER 3 - SWE-Bench Pro 51.3%, fast/cheap
    "llama-nemotron-super-49b", # Backup (v1.5)
    "llama-3.1-70b",            # Backup
]

# Manual fallback with per-model cooldown (avoids LiteLLM's buggy streaming fallback)
COOLDOWN_SECONDS = 30
PER_MODEL_TIMEOUT = 30  # hard timeout per model (connection + initial response)
_cooldowns = {}  # model_name -> cooldown_until_timestamp

def _is_cooled_down(model_name: str) -> bool:
    """True if the model is ready to be tried (cooldown has expired)."""
    return time.time() >= _cooldowns.get(model_name, 0)

def _set_cooldown(model_name: str, seconds: int = COOLDOWN_SECONDS):
    """Mark a model as rate-limited for `seconds`."""
    _cooldowns[model_name] = time.time() + seconds
    logger.info(f"[cooldown] {model_name} cooled down for {seconds}s")

def _build_fallback_chain(model_name: str) -> list:
    """Return the priority-ordered list of models to try, starting from model_name.
    The last model in the chain is always tried even if in cooldown."""
    if model_name in MODEL_PRIORITY:
        idx = MODEL_PRIORITY.index(model_name)
        return MODEL_PRIORITY[idx:]
    return list(MODEL_PRIORITY)  # unknown model, try full chain


def load_env_file():
    """Load .env if present"""
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def build_model_list(api_key: str) -> list:
    """Build model list for LiteLLM Router"""
    return [
        {"model_name": "minimax-m3",                 "litellm_params": {"model": "nvidia_nim/minimaxai/minimax-m3",                             "api_key": api_key}},
        {"model_name": "deepseek-v4-pro",            "litellm_params": {"model": "nvidia_nim/deepseek-ai/deepseek-v4-pro",                      "api_key": api_key}},
        {"model_name": "kimi-k2.6",                 "litellm_params": {"model": "nvidia_nim/moonshotai/kimi-k2.6",                             "api_key": api_key}},
        {"model_name": "nemotron-3-ultra-550b",      "litellm_params": {"model": "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",                 "api_key": api_key}},
        {"model_name": "qwen3.5-397b",               "litellm_params": {"model": "nvidia_nim/qwen/qwen3.5-397b-a17b",                           "api_key": api_key}},
        {"model_name": "nemotron-3-super-120b",      "litellm_params": {"model": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",                 "api_key": api_key}},
        {"model_name": "qwen3-coder-480b",           "litellm_params": {"model": "nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct",              "api_key": api_key}},
        {"model_name": "qwen3-235b",                 "litellm_params": {"model": "nvidia_nim/qwen/qwen3-235b-a22b",                             "api_key": api_key}},
        {"model_name": "deepseek-v4-flash",          "litellm_params": {"model": "nvidia_nim/deepseek-ai/deepseek-v4-flash",                    "api_key": api_key}},
        {"model_name": "glm-5.1",                    "litellm_params": {"model": "nvidia_nim/z-ai/glm-5.1",                                    "api_key": api_key}},
        {"model_name": "step-3.7-flash",             "litellm_params": {"model": "nvidia_nim/stepfun-ai/step-3.7-flash",                        "api_key": api_key}},
        {"model_name": "step-3.5-flash",             "litellm_params": {"model": "nvidia_nim/stepfun-ai/step-3.5-flash",                        "api_key": api_key}},
        {"model_name": "llama-nemotron-super-49b",   "litellm_params": {"model": "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",        "api_key": api_key}},
        {"model_name": "llama-3.1-70b",              "litellm_params": {"model": "nvidia_nim/meta/llama-3.1-70b-instruct",                      "api_key": api_key}},
    ]


def create_router() -> Router:
    """Create LiteLLM Router with all NIM models and fallback chain"""
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    model_list = build_model_list(api_key)
    
    r = Router(
        model_list=model_list,
        num_retries=0,              # we handle retries via manual fallback
        allowed_fails=3,
        timeout=30,                 # per-model timeout (belt and suspenders with asyncio.wait_for)
        enable_pre_call_checks=True,
    )
    
    logger.info(f"Router created with {len(model_list)} models")
    logger.info(f"Priority: {' -> '.join(MODEL_PRIORITY)}")
    
    return r


# === LIFESPAN ===

@contextlib.asynccontextmanager
async def lifespan(app):
    global router
    load_env_file()
    
    if not os.environ.get("NVIDIA_NIM_API_KEY"):
        logger.warning("NVIDIA_NIM_API_KEY not set!")
        logger.warning("Create .env: NVIDIA_NIM_API_KEY=nvapi-...")
        logger.warning("Or: set NVIDIA_NIM_API_KEY=nvapi-...")
    
    router = create_router()
    yield


# === FASTAPI APP ===

app = FastAPI(title="NVIDIA NIM Auto-Fallback Router", lifespan=lifespan)


# === ENDPOINT: HEALTH ===

@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "models_count": len(MODEL_PRIORITY),
        "fallback_chain": MODEL_PRIORITY,
    }


# === ENDPOINT: LIST MODELS ===

@app.get("/v1/models")
async def list_models():
    models_data = []
    for i, name in enumerate(MODEL_PRIORITY):
        models_data.append({
            "id": name,
            "object": "model",
            "created": 1710000000,
            "owned_by": "nvidia-nim",
            "priority": i + 1,
        })
    return {"object": "list", "data": models_data}


# === CORE: CHAT COMPLETIONS ===

# OpenAI params to pass through to the model
# NOTE: only *request* params are forwarded. "tool_calls"/"function_call" are
# *response* fields (they belong inside messages), so they are intentionally
# excluded to avoid passing invalid top-level kwargs to acompletion().
PASSTHROUGH_PARAMS = {
    "messages", "temperature", "top_p", "n", "stream",
    "stop", "max_tokens", "presence_penalty", "frequency_penalty",
    "logit_bias", "user", "response_format", "seed",
    "tools", "tool_choice", "parallel_tool_calls",
    "functions",
    "metadata", "store",
}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Chat completions endpoint — CODING AGENT-READY.
    - Auto-fallback between models
    - Tool/function calling passthrough
    - Streaming passthrough
    - OpenAI-compatible response (identical to direct model call)
    """
    global router
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Map "auto" / "best-available" to a model, preferring the last
    # successful model if it's still in the top tier (sticky routing).
    global _last_successful_model
    model_name = body.get("model", MODEL_PRIORITY[0])
    if model_name in ("auto", "best-available", "best"):
        if _last_successful_model is not None:
            model_name = _last_successful_model
            logger.info(f"[sticky] reusing last-successful model: {model_name}")
        else:
            model_name = MODEL_PRIORITY[0]
    
    # Extract only supported params to pass to the model
    kwargs = {k: v for k, v in body.items() if k in PASSTHROUGH_PARAMS}
    kwargs["model"] = model_name
    stream = kwargs.get("stream", False)
    
    logger.info(f"[{model_name}] {len(kwargs.get('messages', []))} msgs | tools={bool(kwargs.get('tools'))} | stream={stream}")

    if stream:
        # Streaming: run the router call INSIDE the generator so that any
        # failure (even before the first chunk) is delivered as a valid SSE
        # event instead of breaking the text/event-stream contract.
        return StreamingResponse(
            _stream_with_fallback(router, kwargs, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Manual fallback: try models in priority order, stop on first success
    chain = _build_fallback_chain(model_name)
    last_error = None
    for i, model in enumerate(chain):
        # Always try the last model even if in cooldown
        if not _is_cooled_down(model) and i < len(chain) - 1:
            continue
        kwargs["model"] = model
        try:
            response = await asyncio.wait_for(
                router.acompletion(**kwargs), timeout=PER_MODEL_TIMEOUT
            )
            _last_successful_model = model
            return JSONResponse(content=_serialize_response(response))
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"[{model}] timeout ({PER_MODEL_TIMEOUT}s)")
        except Exception as e:
            last_error = e
            err_type = type(e).__name__
            logger.warning(f"[{model}] {err_type}: {str(e)[:200]}")
            if getattr(e, 'status_code', None) == 429:
                _set_cooldown(model)
            # Continue to next model in chain

    err_type = type(last_error).__name__ if last_error else "Unknown"
    logger.error(f"All models failed. Last: {err_type}")
    return JSONResponse(status_code=503, content={
        "error": {
            "message": f"All models failed. Last error: {err_type}",
            "type": err_type,
            "code": "all_models_failed",
        }
    })


@app.post("/chat/completions")
async def chat_completions_alt(request: Request):
    """Alias without /v1/ for compatibility"""
    return await chat_completions(request)


# === HELPERS ===

async def _stream_with_fallback(router, kwargs, model_name):
    """
    SSE streaming with manual fallback — NO LiteLLM fallback chain.
    Tries models in priority order.  Once the first chunk is yielded the
    response is locked in — mid-stream errors on the same model are emitted
    as SSE error events (no cross-model fallback, which would corrupt output).
    """
    global _last_successful_model
    chain = _build_fallback_chain(model_name)
    last_error = None

    for i, model in enumerate(chain):
        # Always try the last model even if in cooldown
        if not _is_cooled_down(model) and i < len(chain) - 1:
            continue
        kwargs["model"] = model
        try:
            response = await asyncio.wait_for(
                router.acompletion(**kwargs), timeout=PER_MODEL_TIMEOUT
            )
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"[{model}] stream connect timeout ({PER_MODEL_TIMEOUT}s)")
            continue
        except Exception as e:
            last_error = e
            err_type = type(e).__name__
            logger.warning(f"[{model}] stream connect {err_type}: {str(e)[:200]}")
            if getattr(e, 'status_code', None) == 429:
                _set_cooldown(model)
            continue  # next model

        # Got a streaming response — iterate chunks
        started_streaming = False
        try:
            async for chunk in response:
                started_streaming = True
                chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
                yield f"data: {json.dumps(chunk_dict, default=str)}\n\n"
            # Stream completed cleanly
            _last_successful_model = model
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            err_type = type(e).__name__
            if started_streaming:
                # Can't fallback mid-stream — output is already partially sent
                logger.error(f"[{model}] mid-stream {err_type}: {str(e)[:200]}")
                err = {"error": {
                    "message": f"Stream interrupted: {err_type}",
                    "type": err_type,
                    "code": "stream_interrupted",
                }}
                yield f"data: {json.dumps(err, default=str)}\n\n"
                yield "data: [DONE]\n\n"
                return
            else:
                # Failed before any chunks — try next model
                last_error = e
                logger.warning(f"[{model}] stream early {err_type}: {str(e)[:200]}")
                if getattr(e, 'status_code', None) == 429:
                    _set_cooldown(model)
                continue

    # All models exhausted
    err_type = type(last_error).__name__ if last_error else "Unknown"
    logger.error(f"All models failed streaming. Last: {err_type}")
    err = {"error": {
        "message": f"All models failed. Last error: {err_type}",
        "type": err_type,
        "code": "all_models_failed",
    }}
    yield f"data: {json.dumps(err, default=str)}\n\n"
    yield "data: [DONE]\n\n"


def _serialize_response(response) -> dict:
    """
    Serialize a LiteLLM response object into a plain dictionary.
    LiteLLM returns ModelResponse objects that already have the correct
    OpenAI structure (choices, usage, model, tool_calls, function_call, etc.).
    """
    if hasattr(response, 'model_dump'):
        return json.loads(response.model_dump_json())
    if hasattr(response, 'dict'):
        return response.dict()
    if isinstance(response, dict):
        return response
    return json.loads(json.dumps(response, default=str))


# === MAIN ===

if __name__ == "__main__":
    # Force UTF-8 encoding on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    
    print("=" * 60, flush=True)
    print("  nim-smart-router", flush=True)
    print("  NIM Auto-Fallback Proxy  |  model: auto  |  coding-agent ready", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print(f"  Server: http://127.0.0.1:4000", flush=True)
    print(f"  Endpoint: POST /v1/chat/completions", flush=True)
    print(f"  Models (by priority):", flush=True)
    for i, m in enumerate(MODEL_PRIORITY, 1):
        print(f"    {i}. {m}", flush=True)
    print(flush=True)
    print("  [V] Tool/function calling passthrough", flush=True)
    print("  [V] Real SSE streaming", flush=True)
    print("  [V] Sticky routing (remembers last successful model)", flush=True)
    print("  [V] Auto-fallback on 429 / errors", flush=True)
    print("  [V] Graceful shutdown (Ctrl+C / window close)", flush=True)
    print("  [V] Standard OpenAI response format", flush=True)
    print("  [V] Compatible with Claude Code, Cursor, Windsurf, Pi", flush=True)
    print(flush=True)
    
    # Load .env BEFORE checking the key, otherwise the warning below always
    # fires (the .env is otherwise only read later, inside lifespan()).
    load_env_file()

    if not os.environ.get("NVIDIA_NIM_API_KEY"):
        print("  [!] NVIDIA_NIM_API_KEY not set!", flush=True)
        print("     Create .env: NVIDIA_NIM_API_KEY=nvapi-...", flush=True)
        print(flush=True)
    
    # === Graceful shutdown ===
    # Ensure the server stops cleanly when the terminal is closed
    # (Ctrl+C, closing the window, or kill signal).
    _shutdown_flag = [False]  # mutable container to avoid global/nonlocal

    def _on_shutdown(signum=None, frame=None):
        if _shutdown_flag[0]:
            return  # already shutting down
        _shutdown_flag[0] = True
        print("", flush=True)
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    atexit.register(lambda: print("\n  [✓] nim-smart-router stopped.", flush=True)
                    if not _shutdown_flag[0] else None)

    signal.signal(signal.SIGINT, _on_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, _on_shutdown)  # kill / window close
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _on_shutdown)  # terminal close (Unix)

    try:
        uvicorn.run(app, host="127.0.0.1", port=4000, log_level="info")
    except KeyboardInterrupt:
        pass  # handled by signal above
    except Exception as e:
        print(f"  [X] {e}", flush=True)
        sys.exit(1)
