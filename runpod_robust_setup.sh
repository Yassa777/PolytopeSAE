#!/bin/bash

# 🚀 ROBUST RUNPOD SETUP SCRIPT
# Handles network timeouts and provides multiple installation strategies

set -e  # Exit on any error

echo "🚀 Starting Robust RunPod Setup for PolytopeSAE"
echo "================================================"

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

# Function to install packages with retries and fallbacks
install_with_retries() {
    local package_list="$1"
    local max_attempts=3
    local timeout=300
    
    for attempt in $(seq 1 $max_attempts); do
        log_info "Installation attempt $attempt/$max_attempts"
        
        # Try with increased timeout and retries
        if pip install --timeout $timeout --retries 5 $package_list; then
            log_success "Installation successful on attempt $attempt"
            return 0
        else
            log_warning "Attempt $attempt failed, trying fallback strategies..."
            
            # Fallback 1: Use different index
            if pip install --timeout $timeout --retries 5 -i https://pypi.python.org/simple/ $package_list; then
                log_success "Installation successful with PyPI fallback"
                return 0
            fi
            
            # Fallback 2: Install without cache
            if pip install --timeout $timeout --retries 5 --no-cache-dir $package_list; then
                log_success "Installation successful without cache"
                return 0
            fi
            
            # Fallback 3: Try conda if available
            if command -v conda &> /dev/null; then
                log_info "Trying conda installation as fallback..."
                if conda install -y $package_list; then
                    log_success "Installation successful with conda"
                    return 0
                fi
            fi
            
            if [ $attempt -lt $max_attempts ]; then
                log_warning "Waiting 30 seconds before retry..."
                sleep 30
            fi
        fi
    done
    
    log_error "All installation attempts failed"
    return 1
}

# 1. Update system and pip
log_info "Updating system and pip..."
pip install --upgrade pip setuptools wheel || {
    log_warning "Pip upgrade failed, continuing anyway..."
}

# 2. Clone repository
log_info "Cloning PolytopeSAE repository..."
if [ ! -d "PolytopeSAE" ]; then
    git clone https://github.com/Yassa777/PolytopeSAE.git || {
        log_error "Failed to clone repository"
        exit 1
    }
fi

cd PolytopeSAE
log_success "Repository cloned and entered"

# 3. Install core dependencies first (most likely to timeout)
log_info "Installing core ML dependencies..."
core_deps="torch>=2.1.0 transformers>=4.41.0 accelerate>=0.28.0"
install_with_retries "$core_deps" || {
    log_error "Failed to install core dependencies"
    exit 1
}

# 4. Install additional ML dependencies
log_info "Installing additional ML dependencies..."
ml_deps="sentencepiece>=0.1.99 safetensors>=0.4.3 huggingface_hub>=0.23.0 tokenizers>=0.13.0"
install_with_retries "$ml_deps" || {
    log_warning "Some ML dependencies failed, continuing..."
}

# 5. Install scientific computing dependencies
log_info "Installing scientific computing dependencies..."
sci_deps="numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.3.0 pandas>=2.0.0"
install_with_retries "$sci_deps" || {
    log_warning "Some scientific dependencies failed, continuing..."
}

# 6. Install remaining dependencies in chunks
log_info "Installing remaining dependencies..."
other_deps="h5py>=3.8.0 wandb>=0.15.0 nltk>=3.8.0 tqdm>=4.65.0 pyyaml>=6.0"
install_with_retries "$other_deps" || {
    log_warning "Some other dependencies failed, continuing..."
}

# 7. Install the package itself
log_info "Installing PolytopeSAE package..."
pip install -e . || {
    log_error "Failed to install PolytopeSAE package"
    exit 1
}

# 8. Download NLTK data
log_info "Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️  NLTK download failed: {e}')
" || log_warning "NLTK download failed, continuing..."

# 9. Set up environment variables
log_info "Setting up environment variables..."
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export WANDB_PROJECT=polytope-hsae" >> ~/.bashrc
echo "export HF_HOME=/workspace/cache/huggingface" >> ~/.bashrc
echo "export TORCH_HOME=/workspace/cache/torch" >> ~/.bashrc

# Create cache directories
mkdir -p /workspace/cache/huggingface /workspace/cache/torch

# 10. Test installation
log_info "Testing installation..."
python -c "
import torch
import transformers
import wandb
import nltk
from polytope_hsae.activations import ActivationCapture
from polytope_hsae.models import HierarchicalSAE
print('✅ All imports successful')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
" || {
    log_error "Installation test failed"
    exit 1
}

# 11. Run quick dry run test
log_info "Running quick dry-run test..."
python experiments/phase1_teacher_extraction.py \
    --config configs/v2-focused.yaml \
    --device cpu \
    --dry-run || {
    log_error "Dry run test failed"
    exit 1
}

log_success "🎉 RunPod setup completed successfully!"
echo ""
echo "📋 NEXT STEPS:"
echo "=============="
echo "1. Set your W&B API key:"
echo "   export WANDB_API_KEY='your_key_here'"
echo ""
echo "2. Set your HuggingFace token (for Gemma access):"
echo "   export HF_TOKEN='your_token_here'"
echo ""
echo "3. Run the full experiment:"
echo "   python experiments/run_all_phases.py \\"
echo "     --config configs/v2-focused.yaml \\"
echo "     --device cuda:0 \\"
echo "     --non-interactive"
echo ""
echo "🚀 Ready for 25-30 hour GPU experiment!"