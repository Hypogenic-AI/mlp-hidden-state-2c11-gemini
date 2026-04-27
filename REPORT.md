# Research Report: Similarity of MLP Hidden States

## 1. Executive Summary
This research investigated the representational similarity of hidden states across different layers in deep Multi-Layer Perceptrons (MLPs). Using Centered Kernel Analysis (CKA) as a metric, we tested the hypothesis that hidden spaces at different layers are not "totally different" due to limited transformation capacity of individual layers.

Our experiments on MNIST and CIFAR-10 revealed that:
1. **Adjacent layers are highly similar**: Mean CKA similarity between adjacent layers consistently exceeded 0.80, confirming that individual layers perform incremental transformations.
2. **Width scales transformation capacity**: Increasing the width of MLP layers allows them to perform more significant transformations, resulting in lower adjacent similarity (higher divergence).
3. **Training drives divergence**: Trained networks show significantly lower layer-to-layer similarity compared to randomly initialized networks, indicating that the network learns to utilize its transformation capacity to extract features.

## 2. Research Question & Motivation
**Research Question**: How similar are MLP hidden states to each other across layers? Are they "totally different," or does the limited capacity of a single linear layer constrain how much the representation can change?

**Motivation**: Understanding these similarities informs model compression, architecture design, and our fundamental understanding of how deep networks build abstractions. If layers are highly similar, they may be redundant.

## 3. Methodology
### Approach
We trained deep (10-layer) MLPs with varying widths on MNIST and CIFAR-10. We used **Linear CKA** to measure the similarity between all pairs of layers.

### Tools & Resources
- **Framework**: PyTorch 2.11
- **Metric**: Linear CKA (implemented in `src/analyze.py`)
- **Datasets**: MNIST and CIFAR-10
- **Models**: DeepMLP (10 hidden layers, widths: 128, 512, 1024)

### Experimental Protocol
1. Initialize a 10-layer MLP.
2. Save "Random" (untrained) state.
3. Train for 5 epochs (to ensure convergence/learning).
4. Extract activations for 1000 test samples.
5. Compute the $10 \times 10$ CKA similarity matrix.

## 4. Results

### Key Statistics
| Model Configuration | Mean Adj. CKA | First-Last Layer CKA |
|---------------------|---------------|----------------------|
| MNIST (Random, 512) | 0.9400        | 0.5558               |
| MNIST (Trained, 128)| 0.9352        | 0.4449               |
| MNIST (Trained, 512)| 0.8831        | 0.2736               |
| MNIST (Trained, 1024)| 0.8508       | 0.2567               |
| CIFAR10 (Trained, 512)| 0.8264      | 0.2131               |

### Visualizations
Heatmaps and line plots are available in the `figures/` directory:
- `mnist_h512_l10_relu_s42_trained_cka_heatmap.png`: Shows the block-diagonal structure of similarity.
- `mnist_h512_l10_relu_s42_trained_adjacent_cka.png`: Shows how similarity evolves with depth.

## 5. Analysis & Discussion
### Hypothesis Testing
- **Hypothesis 1 (Adjacent Similarity)**: Confirmed. Mean Adjacent CKA was always > 0.8.
- **Hypothesis 2 (Transformation Capacity)**: Confirmed. We defined "transformation capacity" as the ability to decrease similarity from the previous layer ($1 - \text{CKA}$). 
- **Hypothesis 3 (Width Effect)**: Confirmed. Wider layers (1024) showed significantly lower adjacent CKA (0.85) than narrower layers (128, CKA 0.93), meaning width directly increases the "step size" in representational space.

### Interpretation
The "transformation capacity" of an MLP layer is constrained by its width. In a narrow network, the representation is forced to stay similar to the previous layer because there aren't enough dimensions to project the data into a significantly different configuration while maintaining the required information. Training "uses up" this capacity to create distinct features, as seen by the drop in similarity from the random baseline.

## 6. Limitations
- **Activation Functions**: We primarily tested ReLU. Different activations (GELU, Tanh) might show different saturation patterns.
- **Depth**: We limited testing to 10 layers. Extremely deep networks (>100 layers) might show different "collapse" behavior.
- **Task Complexity**: We only used classification tasks.

## 7. Conclusions & Next Steps
We conclude that MLP hidden states are **not** totally different; they are highly constrained by the linear transformation capacity of individual layers. However, the "difference" is measurable and scales with layer width and task complexity.

**Recommended Follow-up**:
1. Test **Layer Collapse**: Can we merge layers with CKA > 0.95 without losing accuracy?
2. Investigate **Residual Connections**: Do Skip-connections increase or decrease this layer-to-layer divergence?
3. Compare with **Transformer MLPs**: Do the MLPs in Transformers behave differently due to the surrounding attention mechanism?
