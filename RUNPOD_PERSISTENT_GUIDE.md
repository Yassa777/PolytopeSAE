# 🚀 RunPod Persistent Environment Guide

This guide shows you how to set up a **persistent** Python environment on RunPod that survives pod restarts.

## 🎯 The Problem

By default, when you install packages on RunPod and restart your pod, all packages are lost because they're installed in the temporary container filesystem. This guide fixes that by installing everything in the persistent `/workspace` directory.

## 📦 What You Get

- ✅ Persistent virtual environment in `/workspace/polytope_env`
- ✅ Persistent package cache in `/workspace/.cache/pip` 
- ✅ Persistent model cache for HuggingFace/PyTorch
- ✅ Auto-activation on shell startup
- ✅ Quick reinstall scripts for missing packages

## 🚀 Quick Start

### First Time Setup (Run Once)

1. **SSH into your RunPod** and navigate to your workspace:
   ```bash
   cd /workspace
   ```

2. **Copy your project** to the workspace (if not already there):
   ```bash
   # If cloning from git:
   git clone https://github.com/yourusername/Polytopes.git
   
   # Or copy from container filesystem:
   cp -r /root/Polytopes /workspace/
   ```

3. **Run the persistent setup** (takes 10-15 minutes):
   ```bash
   cd /workspace/Polytopes
   bash runpod_persistent_setup.sh
   ```

### After Pod Restart (Every Time)

Simply activate the persistent environment:
```bash
source /workspace/activate_env.sh
# OR the simpler version:
bash /workspace/Polytopes/activate_persistent.sh
```

That's it! Your environment is restored.

## 🔧 Available Scripts

### Core Scripts

| Script | Purpose |
|--------|---------|
| `runpod_persistent_setup.sh` | **Initial setup** - Creates persistent environment |
| `activate_persistent.sh` | **Quick activation** - Activates environment after restart |
| `install_requirements_persistent.sh` | **Package installer** - Installs with persistent cache |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `/workspace/check_env.py` | **Health check** - Verifies environment status |
| `/workspace/quick_reinstall.sh` | **Emergency fix** - Reinstalls core packages |
| `/workspace/activate_env.sh` | **Full activation** - Complete environment setup |

## 📁 Directory Structure

```
/workspace/
├── polytope_env/          # Persistent virtual environment
├── Polytopes/             # Your project code
├── cache/                 # Persistent model cache
│   ├── huggingface/
│   ├── torch/
│   └── transformers/
├── .cache/pip/            # Persistent pip cache
├── nltk_data/             # NLTK data
└── [activation scripts]
```

## 🔍 Troubleshooting

### Packages Missing After Restart?

```bash
# Quick check
python /workspace/check_env.py

# Quick fix
bash /workspace/quick_reinstall.sh
```

### Environment Not Activating?

```bash
# Check if persistent env exists
ls -la /workspace/polytope_env

# If missing, re-run setup
cd /workspace/Polytopes
bash runpod_persistent_setup.sh
```

### Cache Issues?

```bash
# Check cache directories
ls -la /workspace/cache/
ls -la /workspace/.cache/pip/

# Clear cache if needed
rm -rf /workspace/.cache/pip/*
```

## 💡 Pro Tips

1. **Always use the workspace**: Install everything to `/workspace` paths
2. **Use the cache**: All pip installs automatically use persistent cache
3. **Check your environment**: Run `python /workspace/check_env.py` after restart
4. **Auto-activation**: The environment activates automatically in new shells

## 🎯 Common Workflows

### Running Experiments
```bash
# After pod restart
source /workspace/activate_env.sh
cd /workspace/Polytopes

# Run your experiments
python experiments/run_all_phases.py --config configs/v2-focused.yaml
```

### Installing New Packages
```bash
# Activate environment first
source /workspace/activate_env.sh

# Install with persistent cache
pip install new-package

# Or update requirements
bash install_requirements_persistent.sh
```

### Checking Everything Works
```bash
# Health check
python /workspace/check_env.py

# GPU test
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🚨 Important Notes

- **Always activate the environment** before running code
- **All packages install to `/workspace`** - this is what makes them persistent
- **Cache directories speed up reinstalls** - don't delete them
- **The setup takes time** but only needs to run once per pod template

## 🎉 Success Indicators

When everything works, you should see:
- ✅ Virtual environment at `/workspace/polytope_env`  
- ✅ All packages available after restart
- ✅ GPU access working
- ✅ Fast reinstalls (using cache)
- ✅ Auto-activation in new shells

Happy experimenting! 🚀