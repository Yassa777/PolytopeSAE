#!/bin/bash

# 🎯 SIMPLE PERSISTENT ENVIRONMENT ACTIVATOR
# Use this to activate your persistent environment after pod restart

echo "🚀 Activating persistent PolytopeSAE environment..."

# Set workspace paths
export WORKSPACE_ROOT="/workspace"
export VENV_PATH="$WORKSPACE_ROOT/polytope_env"
export PROJECT_PATH="$WORKSPACE_ROOT/Polytopes"

# Check if persistent environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Persistent environment not found!"
    echo "💡 Run: bash runpod_persistent_setup.sh"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set persistent cache locations (crucial for persistence!)
export HF_HOME="$WORKSPACE_ROOT/cache/huggingface"
export TORCH_HOME="$WORKSPACE_ROOT/cache/torch" 
export TRANSFORMERS_CACHE="$WORKSPACE_ROOT/cache/transformers"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/.cache/pip"
export NLTK_DATA="$WORKSPACE_ROOT/nltk_data"

# Set ML environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="polytope-hsae"

# Change to project directory
if [ -d "$PROJECT_PATH" ]; then
    cd "$PROJECT_PATH"
else
    echo "⚠️  Project directory not found at $PROJECT_PATH"
    echo "💡 Make sure to copy your project to /workspace/Polytopes"
fi

echo "✅ Environment activated!"
echo "📍 Directory: $(pwd)"
echo "🐍 Python: $(which python)"

# Quick health check
if python -c "import torch, transformers" 2>/dev/null; then
    echo "✅ Core packages available"
    if python -c "import torch; print('🎮 CUDA:', torch.cuda.is_available())" 2>/dev/null; then
        echo "🚀 Ready for experiments!"
    fi
else
    echo "⚠️  Some packages missing - run quick_reinstall.sh"
fi