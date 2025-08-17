# Polytope Discovery & Hierarchical SAE

# Integration

**Formal Technical Specification — v1.0 (Aug 2025)**

## 0) Purpose & Scope

This specification defines the algorithms, data, models, metrics, controls, and software requirements for
a program that:

```
Validates geometric structure of categorical and hierarchical concepts in LLMs under the causal
metric (final-layer token→logit geometry).
Incorporates that structure into a Hierarchical Sparse Autoencoder (H‑SAE) to improve concept
purity, reduce leakage, and enable precise steering.
Probes whether similar structure persists inside the network (residual stream activations) using
lightweight local charts , escalating to convex estimation only if required.
```
**Out of Scope:** Full SAEBench replication; safety evaluations; instruction-tuned models; multilingual;
non-text modalities.

## 1) Notation & Mathematical Preliminaries

```
Let be vocabulary size; residual stream dimension ; number of concepts , parents ,
children per parent.
Final-layer unembedding matrix:. Row maps residual state to logit
.
Unembedding covariance: computed over rows of (or a whitened estimate with
shrinkage).
Whitening operator: (symmetric PD). The causal inner product is:
```
```
All geometry (angles, lengths, orthogonality) and estimators are computed in whitened space:
.
Parent vector:. Child delta:.
Hierarchical orthogonality claim:.
LDA estimator (binary concept): With class means and within-class covariance ,
direction. In practice, compute in whitened space with diagonal/shrinkage
.
Explained Variance (EV): , where are activations, reconstruction,
mean.
```
#### 1.

#### 2.

#### 3.

- _V d K P_

### Cp

- _U_ ∈R _V_ × _d uv_ ⊤ _h_ ∈R _d_

### ⟨ uv , h ⟩

- Σ∈R _d_ × _d U_
- _W_ =Σ−1/2 ⟨ _x_ , _y_ ⟩ _c_^ :=

### x ⊤Σ−1 y =( Wx )⊤( Wy ).

#### •

### x ~= Wx

- ℓ _p_ ∈R _d δc_ ∣ _p_ :=ℓ _c_ −ℓ _p_
- ⟨ℓ _p_ , _δc_ ∣ _p_ ⟩ _c_ ≈ 0
- _μ_ 1 , _μ_ 0 _Sw_

### w ∝ Sw −1^ ( μ 1 − μ 0 )

### Sw^

- EV= 1 −∥ _H_ − _H_ ˉ∥
    _F_^2

### ∥ H − H ^∥ F^2 H H ^ H ˉ


## 2) Datasets & Concept Sets

### 2.1 Concept Ontology

```
Parents: WordNet synsets (nouns + verbs), approx..
Children: direct hyponyms per parent (limit for tractability).
```
### 2.2 Prompting & Sampling

```
For each concept, generate 64–256 prompts producing contexts that entail/contrast the concept.
Include negative/foil prompts.
Split by concept: 70/15/15 train/val/test; ensure sibling balance.
```
### 2.3 Models

```
Primary: Gemma‑2B‑it or comparable small base; Secondary: LLaMA‑3‑8B for cross‑model
checks.
```
### 2.4 Activations

```
Capture final residual pre‑unembedding for Phase‑1 (logit geometry).
Capture residuals at selected internal layers for Phases 2–4.
```
## 3) Software Architecture

### 3.1 Repos & Modules

```
`` — compute , whitening , causal projections, angles.
`` — LDA (binary/multiclass), ridge-LDA fallback, centroid/delta estimators.
`` — orthogonality, ratio-invariance, set-inclusion controls.
`` — concept curation, splits, prompt templates.
`` — batched activation capture.
`` — SAE, H‑SAE (router + sub‑SAEs), projectors, TopK gating.
`` — training loops, schedules, losses, logging.
`` — residual edits (parent/child), layer hooks, magnitude schedules.
`` — EV, CE proxy, purity, leakage, steering leakage, invariance KL.
`` — angle histograms, intervention tables, invariance curves.
```
### 3.2 Configuration (YAML)

```
run:
seed: 123
device: cuda:
model:
name: gemma-2b-it
layers_probe: [last, 18, 22]
metric:
whitening: unembedding_rows
shrinkage: 0.
concepts:
```
- _P_ ∈[100,300]
- _Cp_ ≤ 8

#### •

#### •

#### •

#### •

- _Lmid_ , _Llate_
- Σ _W_
- • • • • • • • •


```
parents: 200
max_children: 8
prompts_per_concept: 128
estimators:
type: lda
lda:
shrinkage: 0.
class_balance: true
hsae:
parent_latents: 512
child_latents: 64
topk_child: 1
projector_dim: 512
l1: 1e-
biortho: 1e-
causal_ortho: 5e-
schedule:
freeze_steps: 2000
total_steps: 30000
steering:
layers: [last, 22]
magnitudes: [0.5, 1.0, 2.0]
report:
save_dir: runs/exp
```
## 4) Phase‑1: Final‑Layer Geometry (Must‑Pass Gate)

### 4.1 Procedure

```
Compute and : covariance of rows; add shrinkage; obtain.
Estimate parent and child : LDA in whitened space (binary for parents; one‑vs‑rest for
children). Normalize under.
Orthogonality test: compute where.
Intervention test: add to residual before unembedding; measure target vs sibling logit
deltas; collect effect sizes.
Ratio‑invariance: for siblings , check that moving along preserves up to small
KL.
Controls: (a) shuffle unembeddings; (b) random‑parent replacement; (c) label permutation with
matched support size.
```
### 4.2 Metrics & Thresholds

```
Angle: median (IQR reported); two‑sided test vs shuffled.
Invariance KL: median < 0.10; 90% concepts < 0.20.
Intervention effect: target Δ logits ≥ 3× sibling Δ; Cohen’s.
Controls: angles/invariance collapse to near‑chance.
```
### 1. Σ W U W =Σ−1/

### 2. ℓ p^ ℓ c^

### ∥⋅∥ c

### 3. ∠ c^ (ℓ p^ , δc ∣ p^ ) δc ∣ p^ =ℓ c^ −ℓ p^

### 4. + α ℓ p

### 5. ci ℓ p Δ[( ci / cj )]

#### 6.

- > 80 ∘
-
- _d_ >1.
-


### 4.3 Pseudocode

```
U = load_unembedding();Sigma = cov_rows(U, shrink=0.05)
W = invsqrt(Sigma)
forparent p inParents:
X_pos, X_neg = sample_final_residuals(p)
l_p = lda_direction(W@X_pos, W@X_neg).normalize_causal()
for childc inChildren[p]:
l_c = lda_direction(W@X_c_pos, W@X_c_neg).normalize_causal()
delta= l_c - l_p
angle[p,c] = causal_angle(l_p, delta)
eff[p] = intervene_and_measure(alpha_list, l_p)
run_controls()
report(angle, invariance_kl, eff)
```
## 5) Phase‑2: H‑SAE Baseline (Paper‑style)

### 5.1 Architecture

```
Router (Parent SAE): Enc/Dec operating in causal‑whitened basis; outputs parent codes.
Routed Sub‑SAEs (Children): one per parent; TopK=1 gating selects a child latent for each
token/window; learned down/up projectors between residual and child space.
```
### 5.2 Losses

```
Reconstruction:.
Sparsity: (separate weights for parent vs child tiers allowed).
Bi‑orthogonality: per tier (enc‑dec alignment).
Light causal‑orthogonality (optional): during early epochs to stabilize
hierarchy.
```
### 5.3 Training

```
Optimizer: AdamW; cosine LR; gradient clipping.
Data: residual activations from layers ; batches of tokens/windows.
Logging: EV, CE‑proxy, sparsity, usage, temperature of router, gating stats.
```
### 5.4 Evaluation

```
1‑EV, 1‑CE (lower is better); Purity (parent latents fire on parent contexts), Leakage (cross‑parent
activation), Split/Absorb rates for child units.
```
#### •

#### •

- L _rec_ =∥ _H_ − _H_ ^∥ 22
- _λ_ 1 ∥ _Z_ ∥ 1
- _λb_ ∥ _D_ ⊤ _E_ − _I_ ∥ _F_^2
- _λo_ ∑ _c_ (⟨ _dp_ , _dc_ ∣ _p_ ⟩ _c_ )^2

#### •

- _Lmid_ , _Llate_
-

#### •


## 6) Phase‑3: Teacher‑Initialized H‑SAE (Contribution)

### 6.1 Initialization

```
Map final‑layer parent vectors into the layer’s residual basis using model’s cross‑layer linear
maps if available; otherwise initialize in the decoder space and let training adapt.
Parent decoder rows set to normalized (causal norm).
Child decoder rows set to orthogonalized within each parent’s subspace.
```
### 6.2 Schedule

```
Stage A (Stabilize): freeze decoder directions N=2k–5k steps; enable causal‑orthogonality
regularizer; small LR for router.
Stage B (Adapt): unfreeze with reduced ; standard LR; continue sparsity + bi‑ortho.
```
### 6.3 Comparisons & Metrics

```
Baselines: flat SAE; paper H‑SAE (random init).
Targets: Purity/Leakage (+10%/−20%), Steering leakage (−20%), equal or better 1‑EV/1‑CE ,
lower absorb rate.
```
### 6.4 Ablations

```
Remove teacher init; vary TopK (1→2); drop ; randomize projector dims; magnitude scaling of
initial vectors.
```
## 7) Phase‑4: Internal‑Layer Local Charts (Conditional)

### 7.1 Rationale

Only proceed if Phase‑1/2/3 are successful. Goal: determine whether hierarchical orthogonality and
(near‑)simplicial structure persist in residual activations.

### 7.2 Method (lightweight first)

```
For each parent and layer, compute PCA (s=2–8) on parent‑conditioned activations.
Fit LDA child deltas inside this subspace.
Evaluate ratio‑invariance and edge linearity (linearity of logits when moving along candidate
edges).
```
### 7.3 Escalation (only if needed)

```
Support‑function estimation of polytope facets; consensus‑vertex recovery across bootstraps;
report when this provides additional explanatory power vs PCA/LDA.
```
### 7.4 Diagnostics

```
Holonomy test: traverse small loops within local charts while holding parent coordinate fixed;
measure sibling‑ratio drift. Report “flat enough” when negligible.
```
- ℓ _p_
- ℓˉ _p_
- _δc_ ∣ _p_^ =ℓˉ _c_^ −ℓˉ _p_^

#### •

- _λo_^

#### •

#### •

- _λo_

#### •

#### •

#### •

#### •

#### •


## 8) Phase‑5: Steering Experiments

### 8.1 Edit Types

```
Parent steering: add (layer: last and a late residual layer) to drive parent semantics.
Child steering: add to bias toward specific child while preserving parent level.
H‑SAE steering: activate latent(s) directly via encoder override or decoder injection.
```
### 8.2 Tasks & Readouts

```
Probe tasks where manipulated semantics should change (classification logits, template
generation, masked token selection).
Metrics: target success, leakage (non‑target movement), perplexity change, and locality
(span‑limited effects).
```
## 9) Metrics — Formal Definitions

```
Causal angle:.
Ratio‑invariance KL: For sibling logit distributions before/after move, where
is softmax over sibling set.
Intervention effect size: difference in means divided by pooled std (Cohen’s d) between target
and sibling logit deltas.
Purity: mean activation of a latent on in‑parent examples divided by out‑of‑parent; report
AUROC as alternative.
Leakage: cross‑parent activation rate at a fixed sparsity budget.
Split/Absorb: fraction of child latents that (i) split (multi‑child), or (ii) absorb (become
parent‑like); use clustering over decoder rows + usage stats.
Steering leakage: norm of non‑target changes under an edit.
1‑EV, 1‑CE: report as in SAEBench; lower is better.
```
## 10) Controls & Validity Threats

```
Set‑inclusion confound: (a) shuffled unembeddings; (b) random‑parent replacement; (c) label
permutation with matched support.
Overfitting to ontology: hold‑out synsets; cross‑model replication; verbs vs nouns.
Metric dependence: compare causal vs Euclidean to show necessity of whitening.
Data leakage: strict split by synset; no prompt overlap across splits.
Stability: CI via 1000‑sample bootstrap for key angles/effects only if variance is high.
```
## 11) Implementation Requirements

```
Determinism: fixed seeds; deterministic dataloaders; log exact model/hash.
Artifacts: store , vectors , activation shards, SAE/H‑SAE checkpoints, configs.
Logging: JSONL per step; TensorBoard; CSV summaries; Plotly/Matplotlib figures.
Releases: code + configs + small demo data; pretrained checkpoints for baselines and
teacher‑init models.
```
- + _α_ ℓ _p_
- + _βδc_ ∣ _p_
-

#### •

#### •

• ∠ _c_ (^) ( _x_ , _y_ )=arccos(∥ _x_ ⟨∥ _xc_ , _y_ ∥⟩ _yc_ ∥^ _c_ (^) )

- KL( _psib_^ ∥ _psib_ ′^ ) _psib_^
- • • • • • • • • • • •
- _W_ ℓ _p_ ,ℓ _c_
-
-


## 12) Compute, Runtime & Storage

```
Hardware: 1×A100‑40GB or equivalent; fits in 24GB with smaller batch.
Runtime targets: Phase‑1 < 6 GPU‑hours; Phase‑2/3 per layer 10–24 GPU‑hours.
Storage: < 200 GB (activation shards + checkpoints).
```
## 13) Milestones & Gates

```
Gate A (Phase‑1 pass): angles, invariance, interventions meet thresholds; controls fail as
expected.
Gate B (Phase‑2 pass): paper H‑SAE ≥ flat SAE on 1‑EV/1‑CE; purity/leakage improved.
Gate C (Phase‑3 pass): teacher‑init beats paper H‑SAE on purity/leakage and steering leakage.
Gate D (Phase‑4/5): internal‑layer evidence + steering demos ready for write‑up.
```
## 14) Deliverables

```
Technical report sections (Methods, Results, Ablations) with figures and tables.
Public repo with code/configs and small demo artifacts.
Slide deck: motivation → geometry → H‑SAE → steering.
```
## 15) References

```
The Geometry of Categorical & Hierarchical Concepts in LLMs. (Manuscript.)
Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. (Manuscript.)
```
#### • • • • • • • • • •

#### 1.

#### 2.