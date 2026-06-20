# Rate Limiting: Why and How

## Problem

NVIDIA NIM free tier allows **40 requests per minute** across *all* models for a given API key. This is a **global rate limit**, not per-model. Every HTTP call to any NIM endpoint counts toward this budget — including fallback attempts triggered by 429 errors.

### Current behavior (no rate limiter)

```
User: 10 requests/min
                          ┌─ kimi-k2.6 ── 429 ──┐
Each request → tries 1st  ── deepseek ── 429 ──┤  3 NIM calls
  model(s) in sequence    └─ flash ──── 200 ────┘  per user request
                                                    
Total NIM calls: 10 × 3 = 30 of 40 budget used ✅ (still safe)
```

But if a user makes 15 requests/min with heavy fallbacks:

```
15 requests × 3 NIM calls each = 45 calls → 429 on the 41st
                                    ↑
                            The 41st call hits NIM's hard limit
                            even if it should have succeeded
```

The result: **the 429 comes from NIM's global limit, not from the model's individual limit**. Our fallback logic can't help here — there's no other model to try because **all** are rate-limited.

### Why a coding agent makes this worse

Coding agents often make **parallel tool calls**:

```
Agent sends 5 tool calls at once:
  ├── tool_1 ──► kimi ──► 429 ──► deepseek ──► 429 ──► flash ──► 200
  ├── tool_2 ──► kimi ──► 200
  ├── tool_3 ──► kimi ──► 429 ──► deepseek ──► 200
  ├── tool_4 ──► kimi ──► 200
  └── tool_5 ──► kimi ──► 429 ──► deepseek ──► 429 ──► flash ──► 200

That's 12 NIM calls in ~2 seconds = 18% of the entire minute budget gone.
```

After 4-5 bursts like that, the minute budget is exhausted. Subsequent requests get 429 even from models that would normally answer.

## Solution: Sliding Window with Proportional Delay

### Algorithm

```
┌─────────────────────────────────────────────────────┐
│                  Request arrives                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  1. Check sliding window (timestamps of last 60s)    │
│     Count = len(stamps)                               │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    count < 35              count >= 35
    delay = 0               delay = (count - 34) × 0.5s
    (pass through)          (progressive delay)
                                   │
                                   ▼
                        count = 35 → delay 0.5s
                        count = 36 → delay 1.0s
                        count = 37 → delay 1.5s
                        count = 38 → delay 2.0s
                        count = 39 → delay 2.5s
                        count = 40 → delay 3.0s
                        count = 42 → delay 4.0s
                        count ≥ 44 → delay 5.0s (cap)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────┐
│  2. await asyncio.sleep(delay)                        │
│     (only if delay > 0)                               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  3. Call router.acompletion()                         │
│     ├── 200 → add timestamp → return to client       │
│     └── 429 → fallback → add timestamp → retry       │
│                       │                               │
│            Each HTTP call to NIM adds                  │
│            a timestamp to the sliding window           │
└─────────────────────────────────────────────────────┘
```

### Key properties

| Property | Why |
|----------|-----|
| **Sliding window**, not token bucket | You see the *real* budget consumption over the last 60s, including fallback calls |
| **Proportional delay**, not hard rejection | No client ever gets a 429 from the proxy. Requests are delayed, not blocked. |
| **Delay kicks in at 35/min** (soft cap) | Leaves 5 requests of headroom for bursts during delay |
| **Each NIM call counts**, including fallbacks | Every HTTP request to NIM costs budget — this reflects reality |
| **self-resetting** | After 60s of inactivity, the window is empty and delay is zero |

### What the delay feels like to a coding agent

| Scenario | Delay per request | User-visible effect |
|----------|------------------|---------------------|
| 1 req/min (idle agent) | 0s | None |
| 10 req/min (active coding) | 0s | None |
| 20 req/min (heavy tool use) | 0s | None |
| 30 req/min (intense burst) | 0s | None |
| 35 req/min (very busy) | 0.5s | Barely noticeable |
| 38 req/min (aggressive) | 2.0s | Noticeable but tolerable |
| 42 req/min (extreme) | 4.0s | Slow, but no errors |
| ≥ 44 req/min (cap) | 5.0s | Slow, but no errors |

Coding agents spend most of their time generating tokens (thinking), not calling tools. In practice, a typical coding session stays well under 20 req/min. The delay almost never triggers during normal use — it only protects against bursts or aggressive parallel tool calling.

### Why not a simpler approach?

| Approach | Problem |
|----------|---------|
| **Hard limit (semaforo)** | Agent gets 429 → fallback → same 429 on next model → wastes budget |
| **Token bucket** | Doesn't account for fallback calls that consume extra budget |
| **Per-model limiter** | Doesn't solve the global 40/min problem |
| **Leaky bucket** | Drains too slowly for bursty coding agent patterns |

The sliding window with proportional delay is the *only* approach that:
1. Never rejects a request (no 429 from the proxy)
2. Accurately tracks real NIM consumption (including fallbacks)
3. Self-regulates to stay under the 40/min global cap
4. Feels invisible during normal use

## Implementation

~40 lines of code:

```python
class RateLimiter:
    """Sliding-window rate limiter with proportional delay."""
    
    def __init__(self, max_rpm: int = 40, soft_at: int = 35):
        self.max_rpm = max_rpm
        self.soft_at = soft_at
        self._stamps: list[float] = []  # timestamps of NIM calls in last 60s
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if approaching the rate limit, then record the call."""
        async with self._lock:
            now = time.time()
            # Prune stamps older than 60s
            cutoff = now - 60
            self._stamps = [t for t in self._stamps if t > cutoff]
            
            count = len(self._stamps)
            if count >= self.soft_at:
                # Progressive delay: 0.5s per extra call past soft_at
                delay = (count - self.soft_at + 1) * 0.5
                delay = min(delay, 5.0)  # cap at 5s
                await asyncio.sleep(delay)
            
            self._stamps.append(time.time())
```

Usage in `server.py`:

```python
limiter = RateLimiter()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    ...
    # Before calling NIM, wait if needed
    await limiter.acquire()
    response = await router.acompletion(**kwargs)
    ...
```

### Not implemented

This document is a design proposal. The rate limiter is **not implemented** in the current `server.py`. If desired, it should be added as a standalone module (`ratelimit.py`) and integrated into the chat completions endpoint before the `router.acompletion()` call.
