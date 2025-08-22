#!/bin/bash
# RunPod Quick Setup Script for Polytope Discovery & Hierarchical SAE Integration
# Usage: bash runpod_quick_setup.sh

set -e  # Exit on any error

echo "🚀 Starting RunPod Quick Setup for Polytope Discovery & Hierarchical SAE Integration"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Configure environment variables
log_info "Step 1: Configuring Weights & Biases and HuggingFace..."

# Check if tokens are already set, if not prompt for them
if [ -z "$WANDB_API_KEY" ]; then
    echo "Please provide your W&B API key (get from https://wandb.ai/settings):"
    read -r WANDB_API_KEY
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Please provide your HuggingFace token (get from https://huggingface.co/settings/tokens):"
    read -r HF_TOKEN
fi

# Set default project if not specified
if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT="Polytopes"
fi

# Add to bashrc
echo -e "\n# Weights & Biases and HuggingFace configuration" >> ~/.bashrc
echo "export WANDB_API_KEY=\"$WANDB_API_KEY\"" >> ~/.bashrc
echo "export WANDB_PROJECT=\"$WANDB_PROJECT\"" >> ~/.bashrc
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc

# Source for current session
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="$WANDB_PROJECT" 
export HF_TOKEN="$HF_TOKEN"

log_success "Environment variables configured"

# Step 2: Install Python dependencies
log_info "Step 2: Installing Python dependencies..."
pip install -r requirements.txt
log_success "Python dependencies installed"

# Step 3: Login to HuggingFace
log_info "Step 3: Logging into HuggingFace..."
huggingface-cli login --token $HF_TOKEN
log_success "HuggingFace login completed"

# Step 4: Setup HuggingFace cache directory
log_info "Step 4: Setting up HuggingFace cache..."
mkdir -p ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface
echo 'export HF_HOME=~/.cache/huggingface' >> ~/.bashrc
log_success "HuggingFace cache directory configured"

# Step 5: Pre-download Pythia model
log_info "Step 5: Pre-downloading Pythia-410M model (much faster than Gemma)..."
python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Pythia-410M model...')
model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-410m', 
    device_map='cpu',
    cache_dir=os.environ.get('HF_HOME')
)
print('Downloading Pythia-410M tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/pythia-410m', 
    cache_dir=os.environ.get('HF_HOME')
)
print('✅ Pythia-410M model and tokenizer downloaded successfully!')
"
log_success "Pythia-410M model pre-downloaded"

# Step 6: Install package in development mode
log_info "Step 6: Installing polytope_hsae package in development mode..."
pip install -e .
log_success "Package installed in development mode"

# Step 7: Download NLTK data
log_info "Step 7: Downloading NLTK WordNet data..."
python -c "import nltk; nltk.download('wordnet', quiet=True); print('✅ NLTK WordNet downloaded')"
log_success "NLTK data downloaded"

# Step 8: Verify installation
log_info "Step 8: Verifying installation..."
python -c "
import torch
import transformers
import wandb
import nltk
import polytope_hsae
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'W&B version: {wandb.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"
log_success "Installation verification completed"

# Final status
echo ""
echo "=================================================================="
log_success "🎉 RunPod setup completed successfully!"
echo ""
echo "🔧 Next steps:"
echo "   1. Test with dry run: python experiments/phase1_teacher_extraction.py --config configs/dry-run.yaml --dry-run"
echo "   2. Run full experiment: python experiments/run_all_phases.py --config configs/v2-focused.yaml"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/\$(whoami)/Polytopes"
echo "🔗 HuggingFace Cache: \$HF_HOME"
echo ""
echo "Happy experimenting! 🧪✨"