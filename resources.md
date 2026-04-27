# Resources Catalog

## Summary
This document catalogs all resources gathered for the research on the similarity of MLP hidden states.

## Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Similarity of Neural Network Representations Revisited | Kornblith et al. | 2019 | papers/1905.00414_CKA_Kornblith.pdf | Introduces CKA metric. |
| Insights on representational similarity in neural networks | Morcos et al. | 2018 | papers/1806.05759_PWCCA_Morcos.pdf | PWCCA implementation. |
| LaCo: Large Language Model Pruning via Layer Collapse | Yang et al. | 2024 | papers/2402.11187_LaCo_Yang.pdf | Redundancy in LLM layers. |
| Spectral Analysis of Latent Representations | Shenk et al. | 2019 | papers/1907.03128_SpectralAnalysis_Shenk.pdf | Layer Saturation metric. |
| Do Vision Transformers See Like Convolutional Neural Networks? | Raghu et al. | 2021 | papers/2108.08810_ViT_vs_CNN.pdf | Uniform representations in ViT. |
| OrthoRank | Shin et al. | 2025 | papers/2501.07720_OrthoRank_Shin.pdf | Recent LLM similarity analysis. |

## Datasets
Total datasets referenced: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TinyStories (Samples) | HuggingFace | 100 samples | Text Generation | datasets/tinystories_samples.json | Sample data included. |
| CIFAR-10 | Torchvision | 60K images | Classification | datasets/ (instructions) | Standard benchmark. |
| MNIST | Torchvision | 70K images | Classification | datasets/ (instructions) | Simple baseline. |

## Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| SVCCA | github.com/google/svcca | SVCCA/PWCCA impl | code/svcca/ | Google Research reference. |
| CKA (Custom) | - | CKA implementation | code/cka.py | Ready-to-use PyTorch script. |

## Recommendations for Experiment Design

1. **Primary metric**: Use **Linear CKA** (from `code/cka.py`) as it is the current standard for representation similarity.
2. **Datasets**: Start with **MNIST** or **CIFAR-10** to train small MLPs. Use **TinyStories** if exploring Transformer-like MLPs.
3. **Analysis**:
   - Compare layers within the same model (Layer-wise similarity matrix).
   - Compare layers across different models (Initialization consistency).
   - Investigate the effect of width/depth on similarity patterns.
4. **Baseline**: Compare trained models against **randomly initialized models** to distinguish learned structure from architectural bias (as noted in the ViT vs CNN paper).
