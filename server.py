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

# === MODEL CONFIG ===

# Priority-ordered list: trial order on fallback
MODEL_PRIORITY = [
    "kimi-k2.6",                # TIER 1 - Best
    "deepseek-v4-pro",          # TIER 1 - Best
    "deepseek-v4-flash",        # TIER 2 - Fast
    "minimax-m3",               # TIER 2 - Mid
    "glm-5.1",                  # TIER 2 - Mid
    "step-3.5-flash",           # TIER 3 - Fallback
    "step-3.7-flash",           # TIER 3 - Fallback
    "llama-nemotron-super-49b", # Backup
    "llama-3.1-70b",            # Backup
]

# Build fallback chain: kimi-k2.6 -> deepseek-v4-pro -> ... -> llama-3.1-70b
FALLBACKS = []
for i, model in enumerate(MODEL_PRIORITY):
    fallbacks_list = MODEL_PRIORITY[i+1:]
    if fallbacks_list:
        FALLBACKS.append({model: fallbacks_list})
FALLBACKS.append({"*": [MODEL_PRIORITY[-1]]})


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
        {"model_name": "kimi-k2.6",               "litellm_params": {"model": "nvidia_nim/moonshotai/kimi-k2.6",             "api_key": api_key}},
        {"model_name": "deepseek-v4-pro",          "litellm_params": {"model": "nvidia_nim/deepseek-ai/deepseek-v4-pro",      "api_key": api_key}},
        {"model_name": "deepseek-v4-flash",        "litellm_params": {"model": "nvidia_nim/deepseek-ai/deepseek-v4-flash",    "api_key": api_key}},
        {"model_name": "minimax-m3",               "litellm_params": {"model": "nvidia_nim/minimaxai/minimax-m3",             "api_key": api_key}},
        {"model_name": "glm-5.1",                  "litellm_params": {"model": "nvidia_nim/z-ai/glm-5.1",                    "api_key": api_key}},
        {"model_name": "step-3.5-flash",           "litellm_params": {"model": "nvidia_nim/stepfun-ai/step-3.5-flash",        "api_key": api_key}},
        {"model_name": "step-3.7-flash",           "litellm_params": {"model": "nvidia_nim/stepfun-ai/step-3.7-flash",        "api_key": api_key}},
        {"model_name": "llama-nemotron-super-49b", "litellm_params": {"model": "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1", "api_key": api_key}},
        {"model_name": "llama-3.1-70b",            "litellm_params": {"model": "nvidia_nim/meta/llama-3.1-70b-instruct",     "api_key": api_key}},
    ]


def create_router() -> Router:
    """Create LiteLLM Router with all NIM models and fallback chain"""
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    model_list = build_model_list(api_key)
    
    r = Router(
        model_list=model_list,
        fallbacks=FALLBACKS,
        num_retries=2,
        cooldown_time=30,
        allowed_fails=3,
        retry_after=2,
        max_fallbacks=5,
        timeout=60,                # avoid hanging forever on a stuck upstream
        enable_pre_call_checks=True,
        routing_strategy="simple-shuffle",
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

    # Map "auto" / "best-available" to the top-priority model
    model_name = body.get("model", MODEL_PRIORITY[0])
    if model_name in ("auto", "best-available", "best"):
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
            _stream_passthrough(router, kwargs, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await router.acompletion(**kwargs)
        # Non-streaming: pass response as-is (already OpenAI format)
        return JSONResponse(content=_serialize_response(response))

    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)[:300]
        logger.error(f"[{model_name}] {err_type}: {err_msg}")
        
        # Router exhausted all fallbacks
        content = {
            "error": {
                "message": f"All models failed. Last error: {err_type}",
                "type": err_type,
                "code": "all_models_failed",
            }
        }
        return JSONResponse(status_code=503, content=content)


@app.post("/chat/completions")
async def chat_completions_alt(request: Request):
    """Alias without /v1/ for compatibility"""
    return await chat_completions(request)


# === HELPERS ===

async def _stream_passthrough(router, kwargs, model_name):
    """
    Pass-through SSE streaming from LiteLLM Router.
    The router call is performed here so that errors raised before the first
    chunk (e.g. all fallbacks exhausted) are still emitted as SSE events.
    Each chunk is already in the correct OpenAI format, including tool_calls.
    """
    try:
        response = await router.acompletion(**kwargs)
        async for chunk in response:
            # LiteLLM chunks are ModelResponse (or dict) already in OpenAI format
            chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
            yield f"data: {json.dumps(chunk_dict, default=str)}\n\n"
    except Exception as e:
        err_type = type(e).__name__
        logger.error(f"[{model_name}] stream {err_type}: {str(e)[:300]}")
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
    print("  [V] Auto-fallback on 429 / errors", flush=True)
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
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=4000, log_level="info")
    except Exception as e:
        print(f"  [X] {e}", flush=True)
        sys.exit(1)
