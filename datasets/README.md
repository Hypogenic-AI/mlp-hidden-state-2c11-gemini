# Datasets Catalog

This directory contains datasets for the research project.

## 1. TinyStories (Samples)
- **Source**: `roneneldan/TinyStories` (HuggingFace)
- **Description**: A dataset of short stories for training small language models.
- **File**: `tinystories_samples.json` (First 100 samples)
- **Use Case**: Training small MLPs or Transformer blocks for text-like data.

## 2. CIFAR-10
- **Source**: `torchvision.datasets.CIFAR10`
- **Description**: 60,000 32x32 color images in 10 classes.
- **Download Instructions**:
  ```python
  import torchvision
  train_ds = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)
  test_ds = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True)
  ```
- **Use Case**: Standard image classification for MLPs.

## 3. MNIST
- **Source**: `torchvision.datasets.MNIST`
- **Description**: 70,000 28x28 grayscale images of handwritten digits.
- **Download Instructions**:
  ```python
  import torchvision
  train_ds = torchvision.datasets.MNIST(root='./datasets', train=True, download=True)
  ```
- **Use Case**: Simple baseline for MLP hidden state analysis.
