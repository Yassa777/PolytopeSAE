# Polytope Discovery & Hierarchical SAE Integration

A research framework for validating geometric structure of categorical and hierarchical concepts in Large Language Models and incorporating that structure into Hierarchical Sparse Autoencoders (H-SAEs).

## 🎯 Project Overview

This project implements the experimental framework described in the V2 specification for polytope discovery and hierarchical SAE integration. The core claims being validated are:

1. **Final-layer geometry provides a usable teacher**: Categorical concepts form polytopes in the model's final layer under the causal (whitened) metric
2. **Teacher-initialized H-SAEs improve training**: By initializing H-SAE decoders with geometric parent/child directions, we reduce feature leakage and stabilize training
3. **Hierarchical routing is computationally efficient**: Top-K=1 routing efficiently mirrors conceptual hierarchy following a Mixture-of-Experts pattern

## 🏗️ Architecture

The framework consists of several key modules:

- **`geometry.py`**: Causal inner product computations, whitening, polytope analysis
- **`estimators.py`**: LDA-based estimators for concept vectors and deltas  
- **`validation.py`**: Orthogonality tests, ratio-invariance checks, control experiments
- **`concepts.py`**: WordNet concept curation, prompt generation, data splits
- **`activations.py`**: Batched activation capture from transformer models
- **`models.py`**: Standard SAE and Hierarchical SAE implementations
- **`training.py`**: Training loops, schedules, losses
- **`metrics.py`**: Evaluation metrics (EV, CE, purity, leakage, etc.)
- **`visualization.py`**: Plotting and analysis tools

## 🚀 Quick Start on RunPod

### 1. Initial Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd polytope-hsae

# Run the RunPod setup script
bash runpod_setup.sh

# Activate the environment
source /workspace/startup.sh
```

### 2. Run the V2 Focused Experiment

The V2 experiment is designed to run in 25-30 GPU hours with the following phases:

```bash
# Phase 1: Teacher Vector Extraction (2-3 hours)
python experiments/phase1_teacher_extraction.py --config configs/v2-focused.yaml

# Phase 2: Baseline H-SAE Training (8-10 hours)  
python experiments/phase2_baseline_hsae.py --config configs/v2-focused.yaml

# Phase 3: Teacher-Initialized H-SAE Training (12-14 hours)
python experiments/phase3_teacher_hsae.py --config configs/v2-focused.yaml

# Phase 4: Evaluation & Steering (2-3 hours)
python experiments/phase4_evaluation.py --config configs/v2-focused.yaml
```

Or run all phases sequentially:

```bash
python experiments/run_all_phases.py --config configs/v2-focused.yaml
```

## 📊 Expected Results

### Phase 1 Validation Targets
- **Median causal angle**: ≥ 80° between parent vectors and child deltas
- **Controls**: Geometry should collapse with shuffled unembeddings
- **Interventions**: Parent vector additions should shift target logits

### Phase 2 vs Phase 3 Comparison Targets
- **Purity improvement**: +≥10pp with teacher initialization
- **Leakage reduction**: -≥20% with teacher initialization  
- **Steering leakage**: -≥20% reduction
- **Reconstruction parity**: Similar 1-EV and 1-CE metrics

## 🔧 Configuration

The project uses YAML configuration files in the `configs/` directory:

- **`v2-focused.yaml`**: Main configuration for the focused 25-30 hour experiment
- **`neurips-mi-lowcompute.yaml`**: Extended configuration for the full experiment

Key configuration sections:
- `model`: Model selection and layer specification
- `concepts`: Concept ontology parameters (parents, children, prompts)
- `geometry`: Whitening and estimation parameters
- `hsae`: H-SAE architecture configuration
- `training`: Training schedules and hyperparameters
- `eval`: Evaluation metrics and targets

## 📁 Project Structure

```
polytope-hsae/
├── polytope_hsae/           # Main package
│   ├── geometry.py          # Causal geometry operations
│   ├── estimators.py        # LDA concept vector estimation
│   ├── validation.py        # Geometric validation tests
│   ├── concepts.py          # WordNet concept curation
│   ├── activations.py       # Model activation capture
│   ├── models.py           # SAE and H-SAE architectures
│   ├── training.py         # Training loops and schedules
│   ├── steering.py         # Concept steering experiments
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting and analysis
├── configs/                # Configuration files
├── experiments/            # Experiment runner scripts
├── data/                   # Data storage
├── runs/                   # Experiment outputs
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation
└── runpod_setup.sh        # RunPod environment setup
```

## 🔬 Experimental Phases

### Phase 1: Teacher Vector Extraction (2-3 hours)
1. Compute whitening matrix from unembedding
2. Estimate parent vectors (ℓₚ) and child deltas (δc|p) using LDA
3. Validate hierarchical orthogonality: ⟨ℓₚ, δc|p⟩c ≈ 0
4. Run control experiments (shuffled embeddings, random parents, etc.)

### Phase 2: Baseline H-SAE Training (8-10 hours)
1. Train randomly initialized H-SAE for 7,000 steps
2. Log reconstruction, purity, leakage, and usage statistics
3. Establish baseline performance metrics

### Phase 3: Teacher-Initialized H-SAE Training (12-14 hours)
1. Initialize parent decoder rows with teacher vectors ℓₚ
2. Initialize child decoder rows with orthogonalized child deltas
3. Two-stage training:
   - **Stage A (1,500 steps)**: Freeze decoder, enable causal orthogonality loss
   - **Stage B (8,500 steps)**: Unfreeze decoder, standard training
4. Compare against baseline on all metrics

### Phase 4: Evaluation & Steering (2-3 hours)
1. **Ablations**: Euclidean vs causal, Top-K=1 vs 2, no teacher init
2. **Steering**: Parent and child concept interventions
3. **Analysis**: Effect sizes, leakage measurements, success rates

## 📈 Monitoring and Logging

The framework provides comprehensive logging and monitoring:

- **Weights & Biases**: Real-time experiment tracking
- **TensorBoard**: Local metric visualization  
- **JSON logs**: Structured experiment results
- **Checkpoints**: Model state saving for resumption

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View experiment logs
tail -f runs/v2_focused/phase1_teacher_extraction/*.log

# Launch tensorboard
tensorboard --logdir runs/
```

## 🧪 Key Metrics

### Geometric Validation
- **Causal angles**: Between parent vectors and child deltas
- **Orthogonality**: Inner product magnitudes in causal space
- **Control comparisons**: Shuffled vs real geometry

### H-SAE Performance  
- **Reconstruction**: 1-EV (explained variance), 1-CE (cross-entropy)
- **Purity**: Parent latent activation on parent contexts
- **Leakage**: Cross-parent activation rates
- **Steering**: Off-target effects during concept interventions

### Training Dynamics
- **Sparsity**: Activation rates and usage statistics
- **Stability**: Loss curves and gradient norms
- **Efficiency**: Steps to convergence and final performance

## 🔍 Troubleshooting

### Common Issues

**GPU Memory Issues**:
```bash
# Reduce batch size in config
batch_size_acts: 4096  # down from 8192

# Enable gradient checkpointing
gradient_checkpointing: true
```

**CUDA Errors**:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reset CUDA context
export CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES="0"
```

**Dependency Issues**:
```bash
# Reinstall with specific versions
pip install torch==2.0.1 transformers==4.30.2

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall
```

### Debugging Tools

```bash
# Check system resources
python /workspace/system_info.py

# Validate installation
python -c "from polytope_hsae import geometry; print('✅ Import successful')"

# Test model loading
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('google/gemma-2b'); print('✅ Model loaded')"
```

## 📚 References

1. **The Geometry of Categorical & Hierarchical Concepts in LLMs** (Manuscript)
2. **Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures** (Manuscript)  
3. **WordNet**: A Lexical Database for English
4. **Sparse Autoencoders**: Recent developments in mechanistic interpretability

## 🤝 Contributing

This is a research codebase. For questions or contributions:

1. Check existing issues and documentation
2. Follow the code style (black, isort, flake8)
3. Add tests for new functionality
4. Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

🚀 **Ready to discover polytopes in language model representations!**