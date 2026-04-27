# Paper Outline: How Similar are MLP Hidden States to Each Other?

## 1. Title
How Similar are MLP Hidden States to Each Other? Quantifying Transformation Capacity via CKA

## 2. Abstract
- **Context/Problem**: Deep neural networks build abstractions through a series of transformations across layers. However, the extent to which each layer actually modifies the representation remains poorly quantified.
- **Gap/Challenge**: While metrics like CKA exist to compare models, a focused study on the "step size" of representational change in pure MLPs and how it scales with architecture is missing.
- **Our approach**: We investigate the representational similarity across layers in deep MLPs using Centered Kernel Analysis (CKA). We measure "transformation capacity" as the divergence (1 - CKA) between adjacent layers across various widths and datasets.
- **Key results**: Adjacent layers are highly similar (CKA > 0.8), confirming incremental transformations. Wider layers possess higher transformation capacity, leading to greater divergence. Training significantly increases this divergence compared to random initialization.
- **Significance**: Our findings suggest that layer redundancy is an inherent property of MLP architectures, constrained by their width, which has implications for model compression and architecture design.

## 3. Introduction
- **Hook**: Deep learning's success is attributed to hierarchical feature extraction, yet we often treat layers as black-box transformations.
- **Problem importance**: Understanding layer-to-layer similarity is crucial for model pruning and understanding the fundamental limits of neural transformations.
- **Gap**: Lack of systematic study on representational divergence in MLPs and its dependence on width.
- **Contribution**:
    - We quantify the "transformation capacity" of MLP layers using CKA divergence.
    - We demonstrate that adjacent layer similarity is consistently high (>0.8) but scales with width.
    - We show that training utilizes available transformation capacity to drive representational drift.
    - We provide empirical evidence across MNIST and CIFAR-10 datasets.

## 4. Related Work
- **Similarity Metrics**: Kornblith et al. (2019) introduced CKA; Morcos et al. (2018) used PWCCA.
- **Layer Redundancy & Compression**: Yang et al. (2024) explored layer collapse in LLMs.
- **Representation Evolution**: Raghu et al. (2021) compared ViT vs CNN representations.
- **Spectral Analysis**: Shenk et al. (2019) defined layer saturation.

## 5. Method/Approach
- **Problem Formulation**: Measuring similarity $S(L_i, L_j)$ between layers $i$ and $j$.
- **Metric**: Linear Centered Kernel Analysis (CKA).
- **Models**: 10-layer MLPs with widths 128, 512, 1024.
- **Datasets**: MNIST, CIFAR-10.
- **Protocol**: Compare Random vs. Trained states; Extract activations from 1000 test samples.

## 6. Experiments
- **Setup**: PyTorch 2.11, ReLU activation, 5 epochs of training.
- **Main Results**:
    - Table 1: Mean Adj. CKA and First-Last CKA across configurations.
    - MNIST (Random, 512): Adj CKA 0.94.
    - MNIST (Trained, 512): Adj CKA 0.88.
- **Analysis**:
    - Divergence (1 - CKA) as a measure of "step size".
    - Effect of Width: 1024-wide (0.85) vs 128-wide (0.93).
    - Effect of Training: Random (0.94) vs Trained (0.88).

## 7. Discussion
- **Interpretation**: Width constrains the "step size" in representational space.
- **Limitations**: Restricted to ReLU and 10 layers; classification tasks only.
- **Implications**: Potential for layer merging in narrow networks; architectural design should consider width-driven transformation capacity.

## 8. Conclusion
- **Summary**: MLP hidden states are similar but distinct; transformations are incremental and width-dependent.
- **Takeaway**: Individual layers have limited transformation capacity, which is "used up" during training.
- **Future work**: Test on deeper networks, residual connections, and transformer MLPs.
