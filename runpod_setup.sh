#!/bin/bash

# RunPod Setup Script for Polytope Discovery & Hierarchical SAE Integration
# This script sets up the environment on a RunPod GPU instance

set -e  # Exit on any error

echo "🚀 Setting up Polytope H-SAE Research Environment on RunPod..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl htop nvtop tree

# Set up Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Use persistent virtual environment (or create if not exists)
echo "🔧 Setting up persistent virtual environment..."
PERSISTENT_VENV="/workspace/polytope_env"
if [ ! -d "$PERSISTENT_VENV" ]; then
    echo "Creating new persistent environment..."
    python3 -m venv "$PERSISTENT_VENV"
fi
source "$PERSISTENT_VENV/bin/activate"

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone repository if not already present
if [ ! -d "/workspace/polytope-hsae" ]; then
    echo "📥 Cloning repository..."
    cd /workspace
    git clone https://github.com/your-username/polytope-hsae.git
    cd polytope-hsae
else
    echo "📁 Repository already exists, updating..."
    cd /workspace/polytope-hsae
    git pull
fi

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# Install project in development mode
echo "🔨 Installing project in development mode..."
pip install -e .

# Download NLTK data
echo "📖 Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
print('NLTK data downloaded successfully')
"

# Set up directories
echo "📂 Setting up directories..."
mkdir -p /workspace/polytope-hsae/data
mkdir -p /workspace/polytope-hsae/runs
mkdir -p /workspace/polytope-hsae/checkpoints
mkdir -p /workspace/polytope-hsae/logs
mkdir -p /workspace/polytope-hsae/results

# Set up persistent environment variables and cache
echo "🌍 Setting up persistent environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="/workspace/cache/transformers"
export HF_HOME="/workspace/cache/huggingface"
export TORCH_HOME="/workspace/cache/torch"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export NLTK_DATA="/workspace/nltk_data"
export WANDB_PROJECT="polytope-hsae"

# Create persistent cache directories
mkdir -p /workspace/cache/transformers
mkdir -p /workspace/cache/huggingface
mkdir -p /workspace/cache/torch
mkdir -p /workspace/.cache/pip
mkdir -p /workspace/nltk_data

# Set up Weights & Biases (optional)
echo "📊 Setting up Weights & Biases..."
echo "To use W&B logging, run: wandb login"
echo "Your API key can be found at: https://wandb.ai/authorize"

# Create a persistent startup script for future runs
echo "📝 Creating persistent startup script..."
cat > /workspace/startup.sh << 'EOF'
#!/bin/bash
# Activate persistent environment
source /workspace/polytope_env/bin/activate
cd /workspace/polytope-hsae

# Set persistent cache locations
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="/workspace/cache/transformers"
export HF_HOME="/workspace/cache/huggingface" 
export TORCH_HOME="/workspace/cache/torch"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export NLTK_DATA="/workspace/nltk_data"
export WANDB_PROJECT="polytope-hsae"

echo "🎯 Persistent environment activated! Ready for experiments."
echo "💡 Quick start:"
echo "  - Run Phase 1: python experiments/phase1_teacher_extraction.py --config configs/v2-focused.yaml"
echo "  - Run Phase 2: python experiments/phase2_baseline_hsae.py --config configs/v2-focused.yaml"
echo "  - Run Phase 3: python experiments/phase3_teacher_hsae.py --config configs/v2-focused.yaml"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
EOF

chmod +x /workspace/startup.sh

# Test GPU availability
echo "🔍 Testing GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available - check your setup')
"

# Test transformers installation
echo "🤖 Testing transformers installation..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
print('✅ Transformers installation successful')
"

# Create a quick system info script
echo "📊 Creating system info script..."
cat > /workspace/system_info.py << 'EOF'
#!/usr/bin/env python3
import torch
import psutil
import subprocess
import sys

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "GPU info not available"
    except:
        return "nvidia-smi not found"

def main():
    print("🖥️  System Information")
    print("=" * 50)
    
    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # GPU info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n🎯 GPU Details:")
    print(get_gpu_info())
    
    # Python environment
    print(f"\n🐍 Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except:
        print("Transformers: Not installed")

if __name__ == "__main__":
    main()
EOF

chmod +x /workspace/system_info.py

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Guide:"
echo "1. Activate environment: source /workspace/startup.sh"
echo "2. Check system: python /workspace/system_info.py"
echo "3. Run experiments: cd /workspace/polytope-hsae && python experiments/run_all_phases.py"
echo ""
echo "📁 Important directories:"
echo "  - Project: /workspace/polytope-hsae"
echo "  - Data: /workspace/polytope-hsae/data"
echo "  - Results: /workspace/polytope-hsae/runs"
echo "  - Cache: /workspace/cache"
echo ""
echo "🔧 Useful commands:"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - Check disk space: df -h"
echo "  - View logs: tail -f /workspace/polytope-hsae/logs/*.log"
echo ""
echo "🚀 Ready for polytope discovery experiments!"