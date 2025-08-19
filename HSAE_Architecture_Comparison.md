# JAX H-SAE vs PyTorch H-SAE: Architecture Comparison

## 🔍 **Overview**

This document provides a detailed comparison between the JAX-based Hierarchical SAE implementation found in the `JAX_HSAE` directory and our PyTorch-based implementation in the `polytope_hsae` package. Both implementations follow the same fundamental H-SAE design principles but differ significantly in scale, approach, and research focus.

## 🏗️ **Core Architecture Similarities**

Both implementations share the following fundamental design patterns:

1. **Two-tier hierarchy**: Top-level router SAE + expert-specific sub-SAEs
2. **Top-K routing**: Sparse expert selection mechanism
3. **Subspace projections**: Down/up projectors for each expert's local subspace
4. **Separate encoding/decoding**: Distinct encoder and decoder weight matrices
5. **Sparsity enforcement**: L1 penalties on activations
6. **Reconstruction objective**: MSE loss between input and output

## 📊 **Key Architectural Differences**

| Aspect | **JAX Implementation** | **Our PyTorch Implementation** |
|--------|----------------------|--------------------------------|
| **Scale** | Production-scale (16K experts) | Research-focused (80-256 experts) |
| **Top-K** | k=32 (many active experts) | k=8 (fewer active experts) |
| **Subspace Dim** | 4 (very low-dim) | 96 (higher-dim subspaces) |
| **Atoms per Expert** | 16 | 32 (configurable) |
| **Batch Size** | 32,512 (massive) | 8,192 (reasonable for research) |
| **L1 Penalty** | 1e-3 | 1e-3 (same) |
| **Ortho Penalty** | 1e-1 | 1e-3 to 3e-4 (lighter) |

## 🧠 **Detailed Model Architecture**

### **1. Top-Level Router**

#### **JAX Implementation:**
```python
class Autoencoder(eqx.Module):
    encoder: jnp.ndarray  # Shape: (latent_dim, input_dim)
    decoder: jnp.ndarray  # decoder = encoder.T (tied weights)
    bias: jnp.ndarray
    use_bias: bool
    offset: float = 1.0/sqrt(input_dim)
    
    def encode(self, x):
        x = x - self.bias if self.use_bias else x
        codes = self.encoder @ x
        return leaky_offset_relu(codes, offset=self.offset)
```

**Key Features:**
- Tied weights by default (`decoder = encoder.T`)
- Uses `leaky_offset_relu` with learned offset
- Simple bias handling
- Hard-coded offset based on input dimension

#### **Our PyTorch Implementation:**
```python
class HierarchicalSAE(nn.Module):
    def __init__(self, config: HSAEConfig):
        # Parent (router) SAE
        self.parent_encoder = nn.Linear(config.input_dim, config.n_parents, bias=True)
        self.parent_decoder = nn.Linear(config.n_parents, config.input_dim, bias=False)
        
        # Temperature for gumbel softmax (annealed during training)
        self.register_buffer('router_temperature', torch.tensor(config.router_temp_start))
    
    def encode_parent(self, x):
        parent_logits = self.parent_encoder(x)
        if self.training:
            parent_probs = F.gumbel_softmax(parent_logits, tau=self.router_temperature, hard=True)
        # ... Top-K selection
```

**Key Features:**
- Separate encoder/decoder linear layers (untied by default)
- Gumbel softmax for differentiable routing
- Temperature annealing (1.5 → 0.7)
- Explicit decoder bias parameter

### **2. Expert Subspaces**

#### **JAX Implementation:**
```python
class MixtureOfExperts_v2(eqx.Module):
    def __init__(self, input_dim, subspace_dim=4, atoms_per_subspace=16, num_experts=16384, k=32):
        # Very compact subspaces
        self.W_down = initializer(keys[1], (num_experts, subspace_dim, input_dim))
        self.W_up = jnp.transpose(self.W_down, (0, 2, 1))  # Tied projectors
        
        # Expert encoders/decoders
        self.encoder_weights = initializer(keys[2], (num_experts, atoms_per_subspace, subspace_dim))
        self.decoder_weights = jnp.transpose(self.encoder_weights, (0, 2, 1))  # Tied
```

**Architecture:**
- **16,384 experts** with **4D subspaces**
- **16 atoms per expert**
- **Tied projectors**: `W_up = W_down.T`
- **Tied expert weights**: `decoder = encoder.T`

#### **Our PyTorch Implementation:**
```python
class HierarchicalSAE(nn.Module):
    def __init__(self, config: HSAEConfig):
        for _ in range(config.n_parents):  # Typically 80-256
            # Down/up projectors for this parent's subspace
            down_proj = nn.Linear(config.input_dim, config.subspace_dim, bias=False)  # 96D
            up_proj = nn.Linear(config.subspace_dim, config.input_dim, bias=False)
            
            # Child encoder/decoder in subspace
            child_enc = nn.Linear(config.subspace_dim, config.n_children_per_parent, bias=True)  # 32 atoms
            child_dec = nn.Linear(config.n_children_per_parent, config.subspace_dim, bias=False)
```

**Architecture:**
- **80-256 experts** with **96D subspaces**
- **32 atoms per expert**
- **Untied projectors**: Separate up/down projection matrices
- **Untied expert weights**: Separate encoder/decoder matrices

### **3. Routing Strategy**

#### **JAX Implementation:**
```python
def naive_top_k(data, k):
    """Top k implementation built with argmax. Faster for smaller k."""
    def scannable_top_1(carry, unused):
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)
    
    data, (values, indices) = jax.lax.scan(scannable_top_1, data, (), k)
    return values.T.reshape(-1), indices.T.reshape(-1)

# Usage: k=32 experts active simultaneously
top_k_values, top_k_indices = naive_top_k(top_level_latent_codes[None, :], self.k)
```

**Features:**
- **Hard routing**: No temperature or gradients through selection
- **k=32**: Many experts active per token
- **Scan-based**: Efficient JAX implementation

#### **Our PyTorch Implementation:**
```python
def encode_parent(self, x):
    parent_logits = self.parent_encoder(x)
    
    if self.training:
        # Gumbel softmax for differentiable top-k
        parent_probs = F.gumbel_softmax(parent_logits, tau=self.router_temperature, hard=True)
        topk_vals, topk_indices = torch.topk(parent_logits, self.config.topk_parent, dim=-1)
        parent_codes = torch.zeros_like(parent_logits)
        parent_codes.scatter_(-1, topk_indices, 1.0)
    
    return parent_logits, parent_codes

def update_router_temperature(self, step, total_steps):
    """Update router temperature according to schedule."""
    progress = step / total_steps
    temp = self.config.router_temp_start * (1 - progress) + self.config.router_temp_end * progress
    self.router_temperature.fill_(temp)
```

**Features:**
- **Soft routing**: Gumbel softmax with temperature annealing
- **k=8**: Fewer experts active per token (sparser)
- **Temperature schedule**: 1.5 → 0.7 over training

## 🔧 **Technical Implementation Differences**

### **1. Weight Normalization**

#### **JAX Implementation:**
```python
def project_away_grads(grads, model):
    """Project gradients away from current decoder directions."""
    def vector_reject(a, b):
        normed_b = b / jnp.linalg.norm(b)
        return a - (jnp.dot(a, normed_b)) * normed_b
    
    # Handle top level decoder
    normalized_top_decoder = top_decoder_mat / jnp.linalg.norm(top_decoder_mat, axis=0, keepdims=True)
    top_replace_fn = lambda dec: jax.vmap(vector_reject, in_axes=1, out_axes=1)(dec, normalized_top_decoder)
    # ... Apply to expert decoders too

def normalize_decoders(model):
    """Hard normalization after each step."""
    top_replace_fn = lambda dec: dec / jnp.linalg.norm(dec, axis=0, keepdims=True)
    expert_replace_fn = lambda dec: dec / jnp.linalg.norm(dec, axis=1, keepdims=True)
    # ... Apply normalizations

def update_model(model, grads, opt_state, optimizer):
    projected_grads = project_away_grads(grads, model)  # Project first
    updates, new_opt_state = optimizer.update(projected_grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_model = normalize_decoders(new_model)  # Then normalize
    return new_model, new_opt_state
```

**Approach:**
- **Gradient projection**: Projects gradients away from current decoder directions
- **Hard normalization**: Explicit unit norm enforcement after each step
- **Sophisticated**: Maintains geometric constraints during optimization

#### **Our PyTorch Implementation:**
```python
def normalize_decoder_weights(self):
    """Normalize decoder columns to unit norm."""
    if self.config.normalize_decoder:
        with torch.no_grad():
            # Parent decoder
            parent_norms = torch.norm(self.parent_decoder.weight, dim=0, keepdim=True)
            self.parent_decoder.weight.div_(parent_norms + 1e-8)
            
            # Child decoders
            for i in range(self.config.n_parents):
                child_norms = torch.norm(self.child_decoders[i].weight, dim=0, keepdim=True)
                self.child_decoders[i].weight.div_(child_norms + 1e-8)
```

**Approach:**
- **Simple normalization**: Basic L2 norm division
- **Optional**: Controlled by configuration flag
- **Periodic**: Called explicitly during training (not every step)

### **2. Regularization Strategies**

#### **JAX Implementation:**
```python
def cross_orthogonality_penalty(encoder, decoder):
    """Penalize deviation from orthogonality between encoder and decoder."""
    E = encoder.T / (jnp.linalg.norm(encoder.T, axis=0, keepdims=True) + eps)
    D = decoder.T / (jnp.linalg.norm(decoder.T, axis=1, keepdims=True) + eps)
    ED = D @ E
    dim = ED.shape[0]
    return jnp.linalg.norm(ED - jnp.diag(ED)) / (dim**2 - dim)

def gram_matrix_regularizer(weights):
    """Encourage orthogonality within weight matrices."""
    weights = weights / (jnp.linalg.norm(weights, axis=0, keepdims=True) + 1e-6)
    gram_matrix = jnp.dot(weights.T, weights)
    off_diagonal_elements = gram_matrix - jnp.diag(jnp.diag(gram_matrix))
    dim = off_diagonal_elements.shape[0]
    return jnp.sum(off_diagonal_elements ** 2) / (dim**2 - dim)

# In loss function:
cross_ortho_loss = cross_orthogonality_penalty(model.get_top_level_encoder(), model.get_top_level_decoder())
ortho_loss = cross_ortho_loss  # Other regularizers mostly disabled
total_loss = reconstruction_loss + l1_loss + ortho_penalty * ortho_loss
```

**Focus:**
- **Cross-orthogonality**: E^T D should be diagonal
- **Gram matrix penalties**: Encourage orthogonal weight columns
- **High penalty weight**: ortho_penalty = 1e-1

#### **Our PyTorch Implementation:**
```python
def forward(self, x):
    # ... encoding/decoding ...
    
    # Bi-orthogonality penalty
    if self.config.biorth_lambda > 0:
        biorth_penalty = 0
        
        # Parent bi-orthogonality: E^T D ≈ I
        parent_enc_dec = self.parent_encoder.weight @ self.parent_decoder.weight
        parent_identity = torch.eye(self.config.n_parents, device=x.device)
        biorth_penalty += torch.norm(parent_enc_dec - parent_identity, 'fro') ** 2
        
        # Child bi-orthogonality
        for i in range(self.config.n_parents):
            child_enc_dec = self.child_encoders[i].weight @ self.child_decoders[i].weight
            child_identity = torch.eye(self.config.n_children_per_parent, device=x.device)
            biorth_penalty += torch.norm(child_enc_dec - child_identity, 'fro') ** 2
        
        metrics['biorth_penalty'] = self.config.biorth_lambda * biorth_penalty

def compute_causal_orthogonality_penalty(self, geometry):
    """Novel: Compute causal orthogonality penalty: ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0."""
    penalty = 0
    
    for parent_idx in range(self.config.n_parents):
        parent_vector = self.parent_decoder.weight[:, parent_idx]
        
        for child_idx in range(self.config.n_children_per_parent):
            # Get child vector in full space
            child_subspace = self.child_decoders[parent_idx].weight[:, child_idx]
            child_full = self.up_projectors[parent_idx](child_subspace)
            
            # Compute delta: δ_{c|p} = ℓ_c - ℓ_p
            delta = child_full - parent_vector
            
            # Causal inner product penalty
            inner_prod = geometry.causal_inner_product(parent_vector, delta)
            penalty += inner_prod ** 2
    
    return self.config.causal_ortho_lambda * penalty
```

**Focus:**
- **Bi-orthogonality**: E^T D ≈ I (Frobenius norm)
- **Causal orthogonality**: Novel geometric constraint ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0
- **Lower penalty weights**: biorth_lambda = 1e-3, causal_ortho_lambda = 3e-4

### **3. Teacher Initialization**

#### **JAX Implementation:**
```python
def __init__(self, input_dim, subspace_dim, atoms_per_subspace, num_experts, k, use_bias=False, key=None):
    # No teacher initialization - purely learned from scratch
    initializer = jax.nn.initializers.he_uniform(in_axis=-1, out_axis=(-3,-2))
    
    # Load pre-computed bias (geometric median)
    self.bias = jnp.load("/net/projects2/interp/gemma2-2B_sample_geom_median.npy") if use_bias else None
    self.bias = jnp.zeros_like(self.bias) if use_bias else None
    
    # Random initialization for all weights
    self.W_down = initializer(keys[1], (num_experts, subspace_dim, input_dim))
    self.encoder_weights = initializer(keys[2], (num_experts, atoms_per_subspace, subspace_dim))
```

**Approach:**
- **No teacher initialization**: All weights learned from scratch
- **Geometric median bias**: Uses pre-computed dataset statistics
- **He uniform initialization**: Standard random initialization

#### **Our PyTorch Implementation:**
```python
def initialize_from_teacher(self, 
                          parent_vectors: torch.Tensor,
                          child_projectors: List[Tuple[torch.Tensor, torch.Tensor]],
                          geometry=None):
    """Initialize decoder weights from teacher vectors."""
    logger.info("Initializing H-SAE from teacher vectors")
    
    with torch.no_grad():
        # Initialize parent decoder rows with LDA-estimated concept vectors
        self.parent_decoder.weight.copy_(parent_vectors.T)
        
        if geometry is not None:
            # Normalize under causal norm
            for i in range(self.config.n_parents):
                parent_vec = self.parent_decoder.weight[:, i]
                normalized_vec = geometry.normalize_causal(parent_vec)
                self.parent_decoder.weight[:, i] = normalized_vec
        
        # Initialize projectors from SVD of child delta spans
        for i, (down_proj, up_proj) in enumerate(child_projectors):
            if i >= self.config.n_parents:
                break
                
            self.down_projectors[i].weight.copy_(down_proj)
            self.up_projectors[i].weight.copy_(up_proj)
            
            # Initialize child decoder with orthogonal basis in subspace
            nn.init.orthogonal_(self.child_decoders[i].weight)

# Two-stage training schedule
def freeze_decoder_weights(model: HierarchicalSAE):
    """Freeze decoder weights for stabilization phase."""
    model.parent_decoder.weight.requires_grad = False
    for i in range(model.config.n_parents):
        model.child_decoders[i].weight.requires_grad = False
        model.up_projectors[i].weight.requires_grad = False

def unfreeze_decoder_weights(model: HierarchicalSAE):
    """Unfreeze decoder weights for adaptation phase."""
    model.parent_decoder.weight.requires_grad = True
    for i in range(model.config.n_parents):
        model.child_decoders[i].weight.requires_grad = True
        model.up_projectors[i].weight.requires_grad = True
```

**Approach:**
- **Teacher-driven initialization**: Uses LDA-estimated concept vectors
- **Geometric principles**: Causal norm normalization
- **SVD projectors**: Child subspaces from concept delta spans
- **Two-stage training**: Freeze → adapt schedule

## 🎯 **Research Focus Differences**

### **JAX Implementation: Production Scale**

**Objectives:**
- Scale to massive datasets (billions of tokens)
- Achieve high reconstruction quality
- Efficient memory/compute utilization
- Robust training dynamics

**Approach:**
- **Brute force scaling**: 16K+ experts to capture complexity
- **Minimal constraints**: Light regularization, focus on reconstruction
- **Computational efficiency**: JAX optimizations, large batch sizes
- **Empirical validation**: Reconstruction metrics, activation statistics

**Trade-offs:**
- ✅ Handles massive scale
- ✅ Computationally optimized
- ✅ Proven reconstruction quality
- ❌ No interpretability guarantees
- ❌ Black-box expert specialization
- ❌ No geometric principles

### **Our Implementation: Geometric Principles**

**Objectives:**
- Validate specific geometric hypotheses about concept structure
- Achieve interpretable concept representations
- Demonstrate teacher initialization benefits
- Enable precise concept steering

**Approach:**
- **Theory-driven design**: Teacher initialization from geometric analysis
- **Geometric constraints**: Hierarchical orthogonality validation
- **Research-scale focus**: Manageable expert counts for detailed analysis
- **Interpretability first**: Concept purity, leakage analysis, steering experiments

**Trade-offs:**
- ✅ Interpretable concept structure
- ✅ Geometric validation framework
- ✅ Teacher initialization benefits
- ✅ Precise concept steering
- ❌ Limited to research scale
- ❌ More complex training pipeline
- ❌ Requires concept ontology

## 📈 **Configuration Comparison**

### **Scale Differences**
| Parameter | **JAX Config** | **Our V2 Config** | **Ratio** |
|-----------|----------------|-------------------|-----------|
| **Experts** | 16,384 | 80 | 205× larger |
| **Subspace Dim** | 4 | 96 | 24× smaller |
| **Atoms/Expert** | 16 | 32 | 2× smaller |
| **Top-K** | 32 | 8 | 4× larger |
| **Batch Size** | 32,512 | 8,192 | 4× larger |
| **L1 Penalty** | 1e-3 | 1e-3 | Same |
| **Ortho Penalty** | 1e-1 | 1e-3 to 3e-4 | 33-333× lighter |

### **Training Differences**
| Aspect | **JAX Config** | **Our V2 Config** |
|--------|----------------|-------------------|
| **Learning Rate** | Peak: 5e-4, Init: 1e-11 | 3e-4 (constant) |
| **Warmup Steps** | 1,000 | N/A |
| **Norm Clipping** | 0.75 | 1.0 |
| **Epochs** | 4 | Variable (step-based) |
| **Training Stages** | Single-stage | Two-stage (freeze → adapt) |

### **Architecture Choices**
| Component | **JAX Choice** | **Our Choice** | **Rationale** |
|-----------|----------------|----------------|---------------|
| **Weight Tying** | Tied (decoder = encoder.T) | Untied | More flexibility for concept structure |
| **Routing** | Hard top-k | Gumbel softmax | Differentiable for better gradients |
| **Normalization** | Gradient projection | Simple L2 norm | Simpler, more stable |
| **Regularization** | Cross-orthogonality | Bi-orthogonality + Causal | Geometric principles |
| **Initialization** | Random (He uniform) | Teacher vectors | Leverage known structure |

## 🔬 **Key Innovations in Our Approach**

### **1. Causal Geometry Integration**
```python
class CausalGeometry:
    def __init__(self, unembedding_matrix, shrinkage=0.05):
        # Compute whitening matrix W = Σ^{-1/2}
        self.Sigma, self.W = self._compute_whitening_matrix()
    
    def causal_inner_product(self, x, y):
        """⟨x,y⟩_c = x^T Σ^{-1} y = (Wx)^T (Wy)"""
        x_whitened = self.whiten(x)
        y_whitened = self.whiten(y)
        return torch.sum(x_whitened * y_whitened, dim=-1)
```

**Innovation:** All geometric computations use the causal (whitened) inner product, not Euclidean distance.

### **2. Teacher-Driven Initialization**
```python
# Phase 1: Extract teacher vectors from final-layer geometry
parent_vectors = estimator.estimate_parent_vectors(parent_activations)  # LDA in causal space
child_deltas = estimator.estimate_child_deltas(parent_vectors, child_activations)
projectors = estimator.estimate_child_subspace_projectors(parent_vectors, child_deltas, subspace_dim)

# Phase 2: Initialize H-SAE with teacher knowledge
hsae.initialize_from_teacher(parent_vectors, projectors, geometry)
```

**Innovation:** Leverage geometric structure already present in the model, rather than learning from scratch.

### **3. Hierarchical Orthogonality Validation**
```python
def test_hierarchical_orthogonality(self, parent_vectors, child_deltas, threshold_degrees=80.0):
    """Test the claim that ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0."""
    for parent_id, deltas in child_deltas.items():
        parent_vector = parent_vectors[parent_id]
        for child_id, delta in deltas.items():
            angle_rad = self.geometry.causal_angle(parent_vector, delta)
            angle_deg = torch.rad2deg(angle_rad).item()
            angles_deg.append(angle_deg)
    
    return {
        'median_angle_deg': np.median(angles_deg),
        'fraction_above_threshold': np.mean(angles_deg >= threshold_degrees),
        'passes_validation': median_angle >= threshold_degrees
    }
```

**Innovation:** Explicit validation of geometric claims with statistical testing and control experiments.

### **4. Two-Stage Training Schedule**
```python
# Stage A (Stabilize): freeze decoder directions, enable causal orthogonality
freeze_decoder_weights(model)
for step in range(stabilize_steps):
    loss = reconstruction_loss + l1_loss + causal_ortho_penalty
    
# Stage B (Adapt): unfreeze with reduced causal orthogonality, standard training
unfreeze_decoder_weights(model)
for step in range(adapt_steps):
    loss = reconstruction_loss + l1_loss + biorth_penalty
```

**Innovation:** Preserve geometric structure initially, then allow adaptation with constraints.

## 🎯 **Summary**

### **JAX H-SAE: Production-Scale Approach**
- **Philosophy**: "Scale solves everything"
- **Strengths**: Massive capacity, computational efficiency, proven reconstruction
- **Weaknesses**: Black-box experts, no interpretability guarantees
- **Best for**: Large-scale deployment, high-throughput applications

### **Our PyTorch H-SAE: Geometric Principles Approach**
- **Philosophy**: "Structure guides learning"
- **Strengths**: Interpretable concepts, geometric validation, teacher initialization
- **Weaknesses**: Research-scale only, more complex pipeline
- **Best for**: Scientific understanding, interpretability research, concept steering

### **Key Insight**
The fundamental difference is **bottom-up vs top-down**:
- **JAX approach**: Learn hierarchical structure from scratch through scale
- **Our approach**: Leverage hierarchical structure already present in the model

Both are valid and complementary approaches to hierarchical sparse coding in language models! 🚀

---

*This comparison highlights how the same architectural principles can be implemented with very different research philosophies and objectives.*