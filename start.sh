#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "  nim-smart-router  --  NIM Auto-Fallback Proxy"
echo "  Coding-agent ready  |  model: \"auto\"  |  localhost:4000"
echo "================================================================"
echo ""
echo " Models (by priority):"
echo "   1. minimax-m3           2. deepseek-v4-pro        3. kimi-k2.6"
echo "   4. nemotron-3-ultra-550b 5. qwen3.5-397b          6. nemotron-3-super-120b"
echo "   7. qwen3-coder-480b      8. qwen3-235b            9. deepseek-v4-flash"
echo "  10. glm-5.1              11. step-3.7-flash      12. step-3.5-flash"
echo "  13. nemotron-super-49b   14. llama-3.1-70b"
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

echo " Ctrl+C or close this window to stop the server."
echo ""
python3 server.py || {
    echo " [X] pip install litellm fastapi uvicorn"
    exit 1
}
