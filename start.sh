#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "  nim-smart-router  --  NIM Auto-Fallback Proxy"
echo "  Coding-agent ready  |  model: \"auto\"  |  localhost:4000"
echo "================================================================"
echo ""
echo " Models (by priority):"
echo "   1. kimi-k2.6          2. deepseek-v4-pro      3. minimax-m3"
echo "   4. qwen3-coder-480b   5. qwen3-235b           6. deepseek-v4-flash"
echo "   7. glm-5.1            8. step-3.7-flash       9. step-3.5-flash"
echo "  10. nemotron-super-49b 11. llama-3.1-70b"
echo ""

# Load .env (skip blank lines and comments)
if [ -f ".env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        export "$line"
    done < ".env"
fi

if [ -z "$NVIDIA_NIM_API_KEY" ]; then
    echo " [!] NVIDIA_NIM_API_KEY missing"
    echo "     Create .env: NVIDIA_NIM_API_KEY=nvapi-...  (https://build.nvidia.com)"
    echo ""
fi

echo " Starting on http://127.0.0.1:4000/v1/chat/completions"
echo ""
python3 server.py || {
    echo " [X] pip install litellm fastapi uvicorn"
    exit 1
}
