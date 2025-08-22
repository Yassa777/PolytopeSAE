#!/bin/bash

# 📦 PERSISTENT REQUIREMENTS INSTALLER
# Installs packages with persistent cache to survive pod restarts

set -e

echo "📦 Installing requirements with persistent cache..."

# Ensure we're in the persistent environment
if [ -z "$VIRTUAL_ENV" ] || [[ "$VIRTUAL_ENV" != *"/workspace/"* ]]; then
    echo "⚠️  Not in persistent environment! Activating..."
    source /workspace/polytope_env/bin/activate
fi

# Set persistent cache directory
export PIP_CACHE_DIR="/workspace/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# Function to install with persistent cache and retries
install_with_cache() {
    local requirements_file="$1"
    local max_attempts=3
    
    echo "🔄 Installing from $requirements_file (attempt 1/$max_attempts)..."
    
    for attempt in $(seq 1 $max_attempts); do
        if pip install --cache-dir "$PIP_CACHE_DIR" -r "$requirements_file"; then
            echo "✅ Successfully installed requirements from $requirements_file"
            return 0
        else
            echo "❌ Attempt $attempt failed"
            if [ $attempt -lt $max_attempts ]; then
                echo "⏳ Waiting 30 seconds before retry..."
                sleep 30
                echo "🔄 Installing from $requirements_file (attempt $((attempt + 1))/$max_attempts)..."
            fi
        fi
    done
    
    echo "💥 All attempts failed for $requirements_file"
    return 1
}

# Install from requirements files
if [ -f "requirements.txt" ]; then
    install_with_cache "requirements.txt"
elif [ -f "requirements-minimal.txt" ]; then
    echo "💡 Using minimal requirements..."
    install_with_cache "requirements-minimal.txt"
else
    echo "❌ No requirements file found!"
    echo "💡 Installing essential packages manually..."
    pip install --cache-dir "$PIP_CACHE_DIR" torch transformers accelerate wandb numpy scipy scikit-learn pandas nltk
fi

# Install project in development mode
echo "🔨 Installing project in development mode..."
pip install --cache-dir "$PIP_CACHE_DIR" -e .

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import transformers
import wandb
print('✅ Core packages verified')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "🎉 Requirements installation complete!"
echo "💡 Packages are cached in $PIP_CACHE_DIR for future use"