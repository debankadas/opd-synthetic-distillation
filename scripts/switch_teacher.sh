#!/bin/bash

# Quick Switch Script for Teacher Backends
# Usage: source scripts/switch_teacher.sh [openrouter|gguf|hf]

BACKEND="${1:-openrouter}"

case "$BACKEND" in
  openrouter)
    echo "🔄 Switching to OpenRouter (API-based teacher)..."
    sed -i '' 's/^TEACHER_BACKEND=.*/TEACHER_BACKEND=openrouter/' .env
    echo "✅ Teacher backend set to: openrouter"
    echo ""
    echo "Make sure these are set in .env:"
    echo "  - OPENROUTER_API_KEY=sk-or-v1-..."
    echo "  - OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct"
    echo ""
    echo "💰 Cost: ~\$0.50-2 per 1K training steps"
    ;;
    
  gguf)
    echo "🔄 Switching to GGUF (Local quantized teacher)..."
    sed -i '' 's/^TEACHER_BACKEND=.*/TEACHER_BACKEND=gguf/' .env
    echo "✅ Teacher backend set to: gguf"
    echo ""
    echo "Make sure these are set in .env:"
    echo "  - TEACHER_GGUF=bartowski/Qwen2.5-7B-Instruct-GGUF"
    echo "  - TEACHER_PREFERRED_QUANT=Q4_K_M"
    echo ""
    echo "📦 First run will download ~4-8GB to models/gguf/"
    echo "💰 Cost: Free (local inference)"
    ;;
    
  hf)
    echo "🔄 Switching to HuggingFace (Full local teacher)..."
    sed -i '' 's/^TEACHER_BACKEND=.*/TEACHER_BACKEND=hf/' .env
    echo "✅ Teacher backend set to: hf"
    echo ""
    echo "Make sure these are set in .env:"
    echo "  - TEACHER_MODEL_ID=Qwen/Qwen2.5-7B-Instruct"
    echo "  - TEACHER_DTYPE=bf16"
    echo ""
    echo "📦 First run will download ~15GB"
    echo "💻 Requires: 16GB+ RAM, preferably GPU with 8GB+ VRAM"
    echo "💰 Cost: Free (local inference)"
    ;;
    
  *)
    echo "❌ Invalid backend: $BACKEND"
    echo "Usage: source scripts/switch_teacher.sh [openrouter|gguf|hf]"
    exit 1
    ;;
esac

echo ""
echo "Current .env configuration:"
grep "^TEACHER_BACKEND=" .env
