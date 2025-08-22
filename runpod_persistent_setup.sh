#!/bin/bash

# 🚀 PERSISTENT RUNPOD SETUP SCRIPT
# Installs packages persistently to survive pod restarts
# Uses /workspace directory which is persistent on RunPod

set -e  # Exit on any error

echo "🚀 Setting up PERSISTENT Environment for PolytopeSAE on RunPod"
echo "=============================================================="

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

# Essential directories in persistent workspace
WORKSPACE_ROOT="/workspace"
VENV_PATH="$WORKSPACE_ROOT/polytope_env"
PROJECT_PATH="$WORKSPACE_ROOT/Polytopes"
CACHE_PATH="$WORKSPACE_ROOT/cache"
PYTHON_CACHE="$WORKSPACE_ROOT/.cache/pip"

# 1. Setup persistent directories
log_info "Setting up persistent directories..."
mkdir -p "$CACHE_PATH/huggingface"
mkdir -p "$CACHE_PATH/torch"
mkdir -p "$CACHE_PATH/transformers"
mkdir -p "$PYTHON_CACHE"
mkdir -p "$WORKSPACE_ROOT/data"
mkdir -p "$WORKSPACE_ROOT/runs"
mkdir -p "$WORKSPACE_ROOT/checkpoints"

# 2. Create persistent virtual environment
log_info "Creating persistent virtual environment at $VENV_PATH..."
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate the persistent environment
source "$VENV_PATH/bin/activate"
log_success "Activated persistent environment"

# 3. Upgrade pip in persistent environment
log_info "Upgrading pip in persistent environment..."
pip install --cache-dir "$PYTHON_CACHE" --upgrade pip setuptools wheel

# 4. Clone/update project in workspace
log_info "Setting up project in workspace..."
if [ ! -d "$PROJECT_PATH" ]; then
    log_info "Cloning project to workspace..."
    cd "$WORKSPACE_ROOT"
    # If you have your own repo, replace this URL
    git clone https://github.com/yourusername/Polytopes.git || {
        log_warning "Git clone failed, copying from local if available..."
        if [ -d "/root/Polytopes" ]; then
            cp -r /root/Polytopes "$PROJECT_PATH"
        else
            log_error "No project found to copy. Please manually copy your project to $PROJECT_PATH"
            exit 1
        fi
    }
else
    log_info "Project already exists in workspace"
fi

cd "$PROJECT_PATH"

# 5. Function to install packages with retries and persistence
install_persistent() {
    local package_list="$1"
    local max_attempts=3
    local timeout=600
    
    for attempt in $(seq 1 $max_attempts); do
        log_info "Installing attempt $attempt/$max_attempts: $package_list"
        
        # Install with cache to persistent directory
        if pip install --cache-dir "$PYTHON_CACHE" --timeout $timeout --retries 5 $package_list; then
            log_success "Installation successful on attempt $attempt"
            return 0
        else
            log_warning "Attempt $attempt failed, trying fallback strategies..."
            
            # Fallback 1: No cache
            if pip install --no-cache-dir --timeout $timeout --retries 5 $package_list; then
                log_success "Installation successful without cache"
                return 0
            fi
            
            # Fallback 2: Different index
            if pip install --cache-dir "$PYTHON_CACHE" --timeout $timeout --retries 5 -i https://pypi.org/simple/ $package_list; then
                log_success "Installation successful with alternative index"
                return 0
            fi
            
            if [ $attempt -lt $max_attempts ]; then
                log_warning "Waiting 60 seconds before retry..."
                sleep 60
            fi
        fi
    done
    
    log_error "All installation attempts failed for: $package_list"
    return 1
}

# 6. Install core dependencies in chunks to avoid timeouts
log_info "Installing core PyTorch and transformers..."
install_persistent "torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" || {
    log_warning "PyTorch installation failed, trying CPU version..."
    install_persistent "torch torchvision torchaudio"
}

log_info "Installing transformers ecosystem..."
install_persistent "transformers>=4.41.0 accelerate>=0.28.0 sentencepiece>=0.1.99"

log_info "Installing HuggingFace tools..."
install_persistent "huggingface_hub>=0.23.0 tokenizers>=0.13.0 safetensors>=0.4.3 datasets>=2.12.0"

log_info "Installing scientific computing..."
install_persistent "numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.3.0 pandas>=2.0.0"

log_info "Installing configuration and logging..."
install_persistent "pyyaml>=6.0 hydra-core>=1.3.0 wandb>=0.15.0 tensorboard>=2.13.0"

log_info "Installing data and visualization..."
install_persistent "h5py>=3.8.0 zarr>=2.14.0 tqdm>=4.65.0 matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.14.0"

log_info "Installing NLP and analysis tools..."
install_persistent "nltk>=3.8.0 statsmodels>=0.14.0 pingouin>=0.5.0"

log_info "Installing development tools..."
install_persistent "pytest>=7.4.0 pytest-cov>=4.1.0 black>=23.0.0 isort>=5.12.0 flake8>=6.0.0"

log_info "Installing optional visualization tools..."
install_persistent "umap-learn>=0.5.3 bokeh>=3.2.0"

# 7. Install project in development mode
log_info "Installing project in development mode..."
pip install --cache-dir "$PYTHON_CACHE" -e . || {
    log_error "Failed to install project package"
    exit 1
}

# 8. Download NLTK data to persistent location
log_info "Downloading NLTK data to persistent location..."
export NLTK_DATA="$WORKSPACE_ROOT/nltk_data"
mkdir -p "$NLTK_DATA"
python -c "
import nltk
import os
nltk.data.path.append('$NLTK_DATA')
try:
    nltk.download('wordnet', download_dir='$NLTK_DATA', quiet=True)
    nltk.download('omw-1.4', download_dir='$NLTK_DATA', quiet=True)
    nltk.download('punkt', download_dir='$NLTK_DATA', quiet=True)
    print('✅ NLTK data downloaded to persistent location')
except Exception as e:
    print(f'⚠️  NLTK download failed: {e}')
" || log_warning "NLTK download failed, continuing..."

# 9. Create persistent environment activation script
log_info "Creating persistent environment activation script..."
cat > "$WORKSPACE_ROOT/activate_env.sh" << 'EOF'
#!/bin/bash

# Persistent Environment Activation Script for RunPod
export WORKSPACE_ROOT="/workspace"
export VENV_PATH="$WORKSPACE_ROOT/polytope_env"
export PROJECT_PATH="$WORKSPACE_ROOT/Polytopes"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set persistent cache locations
export HF_HOME="$WORKSPACE_ROOT/cache/huggingface"
export TORCH_HOME="$WORKSPACE_ROOT/cache/torch"
export TRANSFORMERS_CACHE="$WORKSPACE_ROOT/cache/transformers"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/.cache/pip"
export NLTK_DATA="$WORKSPACE_ROOT/nltk_data"

# Set GPU and ML variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="polytope-hsae"

# Change to project directory
cd "$PROJECT_PATH"

echo "🎯 Persistent environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Pip location: $(which pip)"
echo "🎮 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Quick status check
if python -c "import torch, transformers; print('✅ Core packages available')" 2>/dev/null; then
    echo "✅ Environment ready for experiments"
else
    echo "⚠️  Some packages may be missing - run setup again if needed"
fi
EOF

chmod +x "$WORKSPACE_ROOT/activate_env.sh"

# 10. Create auto-activation for new shells
log_info "Setting up auto-activation for new shells..."
echo "# Auto-activate persistent environment" >> ~/.bashrc
echo "if [ -f /workspace/activate_env.sh ]; then" >> ~/.bashrc
echo "    source /workspace/activate_env.sh" >> ~/.bashrc
echo "fi" >> ~/.bashrc

# 11. Create environment check script
log_info "Creating environment verification script..."
cat > "$WORKSPACE_ROOT/check_env.py" << 'EOF'
#!/usr/bin/env python3
"""
Environment verification script for persistent RunPod setup
"""
import os
import sys
import subprocess
from pathlib import Path

def check_package(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name):
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'Unknown')
    except ImportError:
        return 'Not installed'

def check_gpu():
    try:
        import torch
        return torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return False, 0

def check_disk_usage():
    result = subprocess.run(['df', '-h', '/workspace'], capture_output=True, text=True)
    return result.stdout

def main():
    print("🔍 PERSISTENT ENVIRONMENT STATUS CHECK")
    print("=" * 50)
    
    # Basic environment info
    print(f"🐍 Python: {sys.version}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🔧 Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not in venv')}")
    
    # Package status
    packages = [
        'torch', 'transformers', 'accelerate', 'wandb', 
        'numpy', 'scipy', 'sklearn', 'pandas', 'nltk'
    ]
    
    print(f"\n📦 PACKAGE STATUS:")
    for pkg in packages:
        version = get_package_version(pkg)
        status = "✅" if check_package(pkg) else "❌"
        print(f"  {status} {pkg}: {version}")
    
    # GPU status
    cuda_available, gpu_count = check_gpu()
    print(f"\n🎮 GPU STATUS:")
    print(f"  CUDA Available: {'✅' if cuda_available else '❌'}")
    if cuda_available:
        print(f"  GPU Count: {gpu_count}")
        import torch
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Cache directories
    print(f"\n💾 CACHE DIRECTORIES:")
    cache_dirs = [
        ('HF_HOME', os.environ.get('HF_HOME', 'Not set')),
        ('TORCH_HOME', os.environ.get('TORCH_HOME', 'Not set')),
        ('TRANSFORMERS_CACHE', os.environ.get('TRANSFORMERS_CACHE', 'Not set')),
        ('NLTK_DATA', os.environ.get('NLTK_DATA', 'Not set')),
    ]
    
    for name, path in cache_dirs:
        exists = Path(path).exists() if path != 'Not set' else False
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
    
    # Disk usage
    print(f"\n💿 DISK USAGE:")
    try:
        disk_info = check_disk_usage()
        print(disk_info)
    except:
        print("  Could not get disk usage info")
    
    print(f"\n🚀 Setup Status: {'✅ READY' if all(check_package(pkg) for pkg in packages[:4]) else '⚠️  INCOMPLETE'}")

if __name__ == "__main__":
    main()
EOF

chmod +x "$WORKSPACE_ROOT/check_env.py"

# 12. Test the installation
log_info "Testing persistent installation..."
python "$WORKSPACE_ROOT/check_env.py"

# 13. Create quick reinstall script
log_info "Creating quick reinstall script for future use..."
cat > "$WORKSPACE_ROOT/quick_reinstall.sh" << 'EOF'
#!/bin/bash
# Quick reinstall script for missing packages

source /workspace/activate_env.sh
cd /workspace/Polytopes

echo "🔄 Quick reinstall of essential packages..."
pip install --cache-dir /workspace/.cache/pip torch transformers accelerate wandb numpy scipy scikit-learn pandas nltk
pip install --cache-dir /workspace/.cache/pip -e .

echo "✅ Quick reinstall complete!"
python /workspace/check_env.py
EOF

chmod +x "$WORKSPACE_ROOT/quick_reinstall.sh"

log_success "🎉 PERSISTENT ENVIRONMENT SETUP COMPLETE!"
echo ""
echo "📋 SUMMARY:"
echo "==========="
echo "✅ Virtual environment: $VENV_PATH"
echo "✅ Project location: $PROJECT_PATH"  
echo "✅ Cache directories: $CACHE_PATH"
echo "✅ Auto-activation enabled"
echo ""
echo "🚀 USAGE:"
echo "========="
echo "• New shell: source /workspace/activate_env.sh"
echo "• Check status: python /workspace/check_env.py"
echo "• Quick reinstall: bash /workspace/quick_reinstall.sh"
echo "• Run experiments: cd /workspace/Polytopes && python experiments/run_all_phases.py"
echo ""
echo "💡 This environment will persist across pod restarts!"
echo "🔄 If packages go missing, just run: bash /workspace/quick_reinstall.sh"