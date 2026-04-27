# Research Plan: How similar are MLP hidden states to each other?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding the similarity between hidden states across layers is crucial for interpreting how neural networks learn and process information. If layers are highly redundant, we can compress models without significant loss. If they are highly distinct, we can better understand the hierarchy of features. This research specifically addresses the "transformation capacity" of individual layers—how much a single layer *can* actually change the representation.

### Gap in Existing Work
Most representation similarity studies (like CKA) focus on comparing different models or Comparing ViTs vs CNNs. There is less focused work on the *rate of change* of representations in pure, deep MLPs and how architectural constraints (like width and depth) impact this rate. We want to quantify the "step size" in representational space between adjacent layers.

### Our Novel Contribution
We will measure the "Representational Divergence" (1 - CKA) across layers in deep MLPs. We will investigate:
1. How divergence changes with depth (does it saturate?).
2. How width affects the ability of a layer to "change" the representation.
3. Whether a trained MLP has significantly different layer-to-layer similarity patterns compared to a random one.

### Experiment Justification
- **Experiment 1: Layer-wise CKA Matrix**: To visualize the overall similarity structure of the network. High diagonal values would support the hypothesis that layers aren't "totally different."
- **Experiment 2: Width/Depth Sensitivity**: To see if wider layers have more "transformation capacity" (higher divergence from previous layer).
- **Experiment 3: Trained vs. Random**: To distinguish between architectural bias and learned representational drift.

---

## Research Question
The hidden spaces of MLPs at different layers are not totally different due to limited individual transformation capacity, but there are measurable differences between them that can be compared.

## Hypothesis Decomposition
1. **Adjacent Layer Similarity**: Adjacent layers in a deep MLP will show high CKA similarity (>0.8).
2. **Transformation Capacity**: Divergence (1 - CKA) will be higher in earlier layers and decrease (saturate) in later layers.
3. **Width Effect**: Increasing layer width will increase the "transformation capacity," leading to lower similarity between adjacent layers (higher divergence).

## Proposed Methodology

### Approach
We will train deep MLPs on the MNIST and CIFAR-10 datasets. We will use Linear CKA as the primary metric to compare hidden states across all pairs of layers.

### Experimental Steps
1. **Model Training**: 
   - Train a 10-layer MLP (width 512) on MNIST.
   - Train a 10-layer MLP (width 512) on CIFAR-10.
   - Keep checkpoints for "untrained" (random) baselines.
2. **Activation Extraction**: Pass a test set (1000 samples) through the models and save the activations for each layer.
3. **Similarity Analysis**:
   - Compute the full CKA similarity matrix for each model.
   - Extract the "Adjacent CKA" (similarity between layer $i$ and $i+1$).
   - Compute "Cumulative CKA" (similarity between layer 1 and layer $i$).
4. **Width/Depth Ablation**:
   - Compare a 512-wide MLP with a 128-wide and 2048-wide MLP.
5. **Visualization**: Plot heatmaps of similarity matrices and line plots of adjacent similarity across depth.

### Baselines
- **Random Initialization**: Compare trained patterns with a randomly initialized (but same architecture) network.
- **Identity Transform**: A theoretical upper bound of similarity (CKA = 1).

### Evaluation Metrics
- **Linear CKA**: Primary similarity metric.
- **Representational Divergence**: $1 - \text{CKA}$.
- **Accuracy**: To ensure the models are actually learning.

### Statistical Analysis Plan
- We will report mean and standard deviation across 3 different random seeds for training.
- We will use t-tests to compare divergence between early and late layers.

## Expected Outcomes
- We expect to see high similarity (>0.9) between adjacent layers in the middle of the network.
- We expect trained networks to show *more* divergence in earlier layers compared to random networks, as they learn to extract features.

## Timeline and Milestones
- **Phase 1-2**: Planning & Setup (Complete)
- **Phase 3**: Implementation (Scripts for training and CKA) - 1 hour
- **Phase 4**: Experiments (Running multiple models) - 1.5 hours
- **Phase 5**: Analysis (Visualization and Interpretation) - 1 hour
- **Phase 6**: Documentation (Final Report) - 30 min

## Potential Challenges
- **GPU memory**: Extracting all activations for 1000 samples across 20 layers might be heavy. We will process in batches or use smaller sample sizes (N=500 is often enough for CKA).
- **Metric Sensitivity**: CKA might be *too* high if we use ReLU (many zeros). We'll test with GeLU or compare with/without activations.

## Success Criteria
- Successful generation of CKA heatmaps for at least 2 architectures and 2 datasets.
- Clear quantification of "Transformation Capacity" as a function of depth and width.
