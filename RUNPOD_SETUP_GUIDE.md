# 🚀 RunPod Setup Guide - Ready for 25-30 Hour Experiment

## 🔥 **Environment Variables for RunPod**

Set these environment variables in your RunPod container:

### **Required W&B Variables:**
```bash
# W&B Authentication
export WANDB_API_KEY="your_wandb_api_key_here"

# W&B Project Configuration  
export WANDB_PROJECT="polytope-hsae"
export WANDB_ENTITY="your_wandb_username_or_team"

# Optional W&B Settings
export WANDB_MODE="online"  # or "offline" for debugging
export WANDB_CACHE_DIR="/workspace/wandb_cache"
export WANDB_CONFIG_DIR="/workspace/wandb_config"
```

### **Hugging Face Variables (for Gemma):**
```bash
# HF Authentication (if needed for gated models)
export HF_TOKEN="your_huggingface_token_here"
export HUGGINGFACE_HUB_CACHE="/workspace/hf_cache"
```

### **System Variables:**
```bash
# CUDA and Memory
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Python Path
export PYTHONPATH="/workspace/PolytopeSAE:$PYTHONPATH"
```

## 🛠️ **RunPod Container Setup Script**

Save this as `runpod_env_setup.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up Polytope SAE Environment on RunPod"

# Set environment variables
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-polytope-hsae}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_TOKEN="${HF_TOKEN}"

# Create workspace directories
mkdir -p /workspace/wandb_cache
mkdir -p /workspace/wandb_config  
mkdir -p /workspace/hf_cache
mkdir -p /workspace/data
mkdir -p /workspace/runs

# Set cache directories
export WANDB_CACHE_DIR="/workspace/wandb_cache"
export WANDB_CONFIG_DIR="/workspace/wandb_config"
export HUGGINGFACE_HUB_CACHE="/workspace/hf_cache"

# Clone and setup project
cd /workspace
git clone https://github.com/Yassa777/PolytopeSAE.git
cd PolytopeSAE

# Install dependencies
pip install -e .

# Test W&B connection
echo "🔍 Testing W&B connection..."
python -c "import wandb; wandb.login(key='${WANDB_API_KEY}'); print('✅ W&B connected successfully!')"

# Test model loading
echo "🔍 Testing Gemma model loading..."
python -c "from transformers import AutoModelForCausalLM; print('✅ Model loading test passed!')"

echo "🎉 Setup complete! Ready for experiment."
```

## 🎯 **Launch Commands**

### **Full 25-30 Hour Experiment:**
```bash
# Set your variables first!
export WANDB_API_KEY="your_key_here"
export WANDB_ENTITY="your_username"
export HF_TOKEN="your_hf_token"  # if needed

# Run complete experiment
python experiments/run_all_phases.py --config configs/v2-focused.yaml --device cuda:0
```

### **Individual Phase Testing:**
```bash
# Test Phase 1 (Teacher Extraction)
python experiments/phase1_teacher_extraction.py --config configs/v2-focused.yaml --device cuda:0

# Test Phase 2 (Baseline H-SAE)  
python experiments/phase2_baseline_hsae.py --config configs/v2-focused.yaml --device cuda:0

# Test Phase 3 (Teacher H-SAE)
python experiments/phase3_teacher_hsae.py --config configs/v2-focused.yaml --device cuda:0

# Test Phase 4 (Evaluation)
python experiments/phase4_evaluation.py --config configs/v2-focused.yaml --device cuda:0
```

### **Dry Run Testing:**
```bash
# Quick test with small model (5 minutes)
python experiments/phase1_teacher_extraction.py --config configs/v2-focused.yaml --dry-run --device cuda:0
```

## 🔧 **W&B Integration Status**

✅ **Phase 1**: Basic logging (teacher vectors, validation results)  
✅ **Phase 2**: Full training logs (loss, metrics, model architecture)  
✅ **Phase 3**: Two-stage training logs (stabilize + adapt phases)  
✅ **Phase 4**: Evaluation metrics, steering results, comparisons  

### **W&B Project Structure:**
- **Project**: `polytope-hsae`
- **Run Names**: `phase{N}_{timestamp}` (e.g., `phase2_baseline_20240115_143022`)
- **Tags**: `['phase1', 'teacher-extraction']`, `['phase2', 'baseline']`, etc.
- **Metrics**: Loss curves, 1-EV, purity, leakage, steering precision
- **Artifacts**: Model checkpoints, teacher vectors, validation results

## 📊 **Expected W&B Dashboards**

### **Phase 2 & 3 Training:**
- **Loss Curves**: Total, reconstruction, L1, bi-orthogonality
- **Metrics**: 1-EV, 1-CE, purity, leakage
- **Sparsity**: Parent/child activation rates, usage statistics  
- **Router**: Temperature annealing, top-K activation patterns
- **Architecture**: Model size, hyperparameters

### **Phase 4 Evaluation:**
- **Comparisons**: Teacher vs Baseline performance
- **Steering**: Precision, leakage, effect sizes
- **Ablations**: Euclidean vs Causal geometry
- **Success Metrics**: Target achievement rates

## 🎯 **What You Need To Provide:**

1. **WANDB_API_KEY**: Get from https://wandb.ai/settings
2. **WANDB_ENTITY**: Your W&B username or team name
3. **HF_TOKEN**: (Optional) From https://huggingface.co/settings/tokens if using gated models

That's it! Everything else is already configured and ready to go! 🚀

## 🔥 **Expected Results in W&B:**

- **4 separate runs** (one per phase)
- **Comprehensive metrics** for teacher vs baseline comparison
- **Model checkpoints** automatically saved
- **Geometric validation** results with angle distributions
- **Steering experiment** results with precision metrics

Your experiment will be **fully tracked and reproducible**! 📊✨