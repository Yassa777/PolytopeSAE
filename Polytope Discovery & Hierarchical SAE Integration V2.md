# **Polytope Discovery & Hierarchical SAE Integration**

## **Formal Technical Specification v2.0 (Aug 2025\)**

### **0\) Purpose & Scope**

This specification defines a focused experiment to prove a set of tight, targeted claims about concept geometry and its application to Hierarchical Sparse Autoencoders (H-SAEs).  
**Core Claims:**

1. **Final-layer geometry provides a usable teacher.** Categorical concepts form polytopes in the model's final layer. Under the causal (whitened) metric, parent concepts are represented by vectors, child concept contrasts are vector differences, and the hierarchy manifests as orthogonality.  
2. **Teacher-initialized H-SAEs improve training.** By initializing an H-SAE's decoder with these geometric parent/child directions, we can significantly reduce feature leakage and stabilize training compared to a randomly initialized H-SAE, while matching reconstruction performance.  
3. **Hierarchical routing is computationally efficient.** By activating a child SAE only when its parent feature fires (Top-K=1 routing), the architecture efficiently mirrors the conceptual hierarchy, following a Mixture-of-Experts (MoE) pattern.

**Out of Scope:** Full SAEBench replication; safety evaluations; instruction-tuned models; multilingual; non-text modalities; SAEs on multiple or interior model layers.

### **1\) Notation & Mathematical Preliminaries**

* Let V be vocabulary size; residual stream dimension d.  
* **Unembedding Matrix:** U∈RV×d.  
* **Unembedding Covariance:** Σ∈Rd×d, computed over rows of U.  
* **Whitening Operator:** W=Σ−1/2. The **causal inner product** is ⟨x,y⟩c​:=x⊤Σ−1y=(Wx)⊤(Wy). All geometry is computed in this whitened space.  
* **Parent Vector:** ℓp​∈Rd.  
* **Child Delta:** δc∣p​:=ℓc​−ℓp​.  
* **Hierarchical Orthogonality Claim:** ⟨ℓp​,δc∣p​⟩c​≈0.  
* **LDA Estimator:** Used to find concept directions in whitened space.

### **2\) Datasets, Models & Activations**

* **Model:** Gemma-2B (base or \-it).  
* **Layers:** All SAEs and geometric analysis will be performed on **one late residual layer** (the "last" layer) to conserve compute.  
* **Concept Ontology:**  
  * **Parents (P):** **80** WordNet synsets (down from 200).  
  * **Children per parent (**Cp​**):** **≤ 4** (capped at the most salient four).  
* **Prompting & Sampling:**  
  * **64** prompts per concept (down from 128).  
  * Split 70/15/15 train/val/test by concept.  
* **Activations:** Capture final residual pre-unembedding for all phases.

### **3\) H-SAE Architecture & Configuration**

* **Architecture:** A router (parent SAE) gates access to child SAEs (one per parent concept) via Top-K=1 routing.  
* **Configuration Deltas:**  
  * parent\_latents: **256** (from 512\)  
  * child\_latents: **32** (from 64\)  
  * projector\_dim: **256** (from 512\)  
  * topk\_child: **1** (unchanged)  
* **Losses:**  
  * Reconstruction: Lrec​=∥H−H^∥22​  
  * Sparsity: λ1​∥Z∥1​  
  * Bi-orthogonality: λb​=5e-4 (down-weighted)  
  * Causal-orthogonality: λo​=2e-4 (applied only during the initial freeze stage of teacher-initialized training).

### **4\) Experimental Plan: A 25-30 GPU-Hour Agenda**

This plan executes two main runs on a single late layer to compare a baseline H-SAE against a teacher-initialized one.

#### **Phase 1: Teacher Vector Extraction (Est. 2–3 hours)**

1. **Compute Whitening Matrix:** Calculate Σ and W from the final layer's unembedding matrix.  
2. **Estimate Vectors:** Use LDA in the whitened causal space to find parent vectors (ℓp​) and child delta vectors (δc∣p​) for the 80 parents and their children.  
3. **Sanity Checks:**  
   * **Angles:** Verify that the median causal angle ∠c​(ℓp​,δc∣p​) is \> 80°.  
   * **Interventions:** Confirm that adding \+αℓp​ to activations shifts parent logits with minimal change to child ratios.  
   * **Controls:** Ensure geometry collapses with shuffled unembeddings, random parent replacement, and label permutations.

#### **Phase 2: Baseline H-SAE Training (Est. 8–10 hours)**

* **Initialization:** Randomly initialize the H-SAE.  
* **Training:** Train for **7,000 steps** with AdamW, bf16, and gradient accumulation to maximize throughput on an A100-40GB GPU.  
* **Evaluation:** Log reconstruction (EV/CE), purity, leakage, and feature usage statistics.

#### **Phase 3: Teacher-Initialized H-SAE Training (Est. 12–14 hours)**

* **Initialization:**  
  * Initialize parent decoder rows with the teacher vectors ℓp​ from Phase 1\.  
  * Initialize child decoder rows with an orthogonalized basis for the child delta vectors δc∣p​.  
* **Training Schedule (10,000 steps total):**  
  1. **Freeze Decoder (1,500 steps):** Train only the router and encoders. The decoder weights are frozen. Enable the light causal-orthogonality loss (λo​=2e-4) to stabilize the geometric structure.  
  2. **Adapt (8,500 steps):** Unfreeze the decoder, disable or reduce the causal-orthogonality loss, and continue training all components with a lower learning rate on the decoder rows.  
* **Evaluation:** Compare against the baseline on all metrics, with a focus on purity and leakage improvements.

#### **Phase 4: Evaluations & Steering (Est. 2–3 hours)**

* **Ablations:**  
  * **Euclidean vs. Causal:** Show that extracting teacher vectors in Euclidean space breaks the geometric properties.  
  * **Top-K=1 vs. 2:** Demonstrate that K=2 increases reconstruction slightly but significantly increases leakage.  
  * **No Teacher:** Confirm that removing teacher initialization degrades purity and leakage back to baseline levels.  
* **Steering Demos:** Perform parent and child concept edits, measuring success rate and **steering leakage** (off-target effects).

### **5\) Theoretical Contributions**

#### **T1. Teacher-Orthogonality ⇒ Block-diagonal Training at Init (First-Order Leak Bound)**

* **Claim:** If an H-SAE decoder is initialized with a parent vector ℓp​ and child rows spanning the causally orthogonal child-delta subspace, the gradient of the reconstruction and bi-orthogonality losses with respect to off-parent child rows is orthogonal to the parent row. Cross-parent leakage is therefore a second-order effect at initialization.  
* **Payoff:** This provides a formal **leakage bound at init**: Leakage ≤O(sinθ), where θ is the deviation from perfect hierarchical orthogonality. It explains *why* teacher initialization rapidly cuts leakage.

#### **T2. Top-K Routing ≈ Normal-Fan of the Polytope (Geometric Interpretation)**

* **Claim:** In the causal space, the Top-K=1 encoder gating mechanism implements a polyhedral partition of the activation space. The boundaries of this partition coincide with the normal fan of the categorical polytope defined by the child concepts.  
* **Payoff:** This connects the SAE's architectural bias (Top-K routing) directly to the underlying geometric structure of the concepts it's meant to represent, explaining why K=1 is effective at isolating individual children.

#### **T3. Sample-Efficiency Argument for Teacher vs. Random Init**

* **Claim:** Initializing decoder rows at an angle ≤ϵ to the true parent vectors improves the sample complexity of learning compared to random initialization by a factor of ∼1/(1−cosϵ).  
* **Payoff:** Frames the teacher-initialization approach not as a mere heuristic but as a principled method for improving generalization and training speed.

### **6\) Publication Strategy & Framing**

* **Target:** NeurIPS-MI or similar venue focused on interpretability and mechanistic understanding.  
* **Framing:** This work aligns an **architectural constraint** (Top-K H-SAE) with **measured final-layer geometry** (polytopes, vector hierarchy). We demonstrate that designing SAEs to match the known structure of concepts—a principle from works like "Projecting Assumptions"—yields measurable gains in feature purity and training stability.  
* **Paper Structure:**  
  1. **Background:** Causal inner product, vector reps, polytopes, orthogonality.  
  2. **Method:** The teacher-initialized H-SAE with its freeze-then-adapt schedule.  
  3. **Theory:** T1 and T2 as formal claims.  
  4. **Experiments:** Phase 1 geometry checks, the core baseline vs. teacher-init comparison, and key ablations.  
  5. **Steering:** A concise demonstration of improved steering with lower leakage.

### **7\) Acceptance-Oriented Targets & Metrics**

* **Geometry (Phase 1):** Median causal angle ∠c​(ℓp​,δc∣p​)≥80∘, with controls collapsing to near-chance.  
* **H-SAE Comparison (Phase 3 vs. 2):**  
  * **Purity:** \+≥10 pp improvement with teacher-init.  
  * **Leakage:** \-≥20% reduction.  
  * **Steering Leakage:** \-≥20% reduction.  
  * **Reconstruction:** Parity on 1-EV and 1-CE metrics.  
* **Ablations:** A clear figure showing geometry breaks under the Euclidean metric is critical for demonstrating the necessity of the causal framework.