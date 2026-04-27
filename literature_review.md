# Literature Review: Similarity of MLP Hidden States

## Research Area Overview
The similarity of hidden states across different layers in neural networks (especially MLPs and Transformers) is a key area of study for understanding how representations evolve during processing. Measuring this similarity helps in model pruning, interpretability, and understanding generalization.

## Key Papers

### 1. Similarity of Neural Network Representations Revisited (CKA)
- **Authors**: Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey E. Hinton
- **Year**: 2019
- **Source**: ICML
- **Key Contribution**: Introduces Centered Kernel Alignment (CKA) as a robust similarity metric for neural representations.
- **Methodology**: CKA compares the similarity of representational similarity matrices (RSMs). It is invariant to orthogonal transformation and isotropic scaling.
- **Results**: CKA reliably identifies correspondences between layers of different models, unlike CCA or other metrics. It shows that representations in the same model are often similar across nearby layers.
- **Relevance**: Provides the primary metric (Linear CKA) for comparing hidden states.

### 2. LaCo: Large Language Model Pruning via Layer Collapse
- **Authors**: Yifei Yang, Zouying Cao, Hai Zhao
- **Year**: 2024
- **Source**: arXiv
- **Key Contribution**: Proposes "Layer Collapse" (LaCo) for pruning LLMs by collapsing rear layers into prior layers.
- **Methodology**: Uses layer-wise similarity (CKA) to justify that many layers are redundant and can be merged.
- **Results**: Shows that many layers in deep LLMs have very high similarity, especially in the later stages of the model.
- **Relevance**: Directly supports the hypothesis that MLP/Transformer layers have high similarity and measurable differences.

### 3. Do Vision Transformers See Like Convolutional Neural Networks?
- **Authors**: Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy
- **Year**: 2021
- **Source**: NeurIPS
- **Key Contribution**: Compares internal representations of ViT and CNN.
- **Results**: ViTs have much more uniform representations across layers compared to CNNs. Residual connections in ViTs play a huge role in propagating features and maintaining similarity.
- **Relevance**: Suggests that MLP-like structures (blocks in Transformers) may maintain higher similarity across layers than traditional CNNs.

### 4. Spectral Analysis of Latent Representations
- **Authors**: Justin Shenk, Mats L. Richter, Anders Arpteg, Mikael Huss
- **Year**: 2019
- **Source**: arXiv
- **Key Contribution**: Defines "Layer Saturation" based on the variance explained by eigenvalues.
- **Relevance**: Provides another perspective on similarity/redundancy—if multiple layers have similar "saturation," they might be performing similar transformations.

### 5. Insights on representational similarity in neural networks with canonical correlation (PWCCA)
- **Authors**: Ari S. Morcos, Maithra Raghu, Samy Bengio
- **Year**: 2018
- **Source**: NeurIPS
- **Key Contribution**: Improves upon SVCCA with Projection Weighted CCA (PWCCA).
- **Results**: Found that networks converge to similar representations if they generalize well.
- **Relevance**: Foundations of using CCA-based metrics for layer comparison.

## Common Methodologies
- **CKA (Centered Kernel Alignment)**: Most popular for comparing across models and layers.
- **Canonical Correlation Analysis (CCA/SVCCA/PWCCA)**: Used for finding linear subspaces that correlate.
- **Model Stitching**: A functional way to measure similarity—if layer A can be swapped with layer B (with a small transformation), they are similar.

## Standard Baselines
- **Linear Probing**: Measuring if the same information can be extracted from different layers.
- **Representation Similarity Matrices (RSMs)**: Computing the pairwise similarity of inputs at each layer.

## Evaluation Metrics
- **Linear CKA Score**: (0 to 1), where 1 is identical representation (up to scaling/rotation).
- **Layer Saturation**: Dimensionality of the representation.
- **Stitching Accuracy**: Performance drop when swapping layers.

## Recommendations for Our Experiment
- **Dataset**: CIFAR-10 (standard), TinyStories (for LLM-like MLPs).
- **Models**: Simple MLPs with varying depths (e.g., 5, 10, 20 layers) and widths.
- **Analysis**:
    1. Compute CKA between every pair of layers (i, j).
    2. Visualize similarity matrices.
    3. Test "Layer Collapse" (swapping/merging layers) to see functional similarity.
    4. Measure Layer Saturation to see if representations "compress" or "expand."
