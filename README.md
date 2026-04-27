# Similarity of MLP Hidden States Across Layers

This project investigates the representational similarity between layers in deep Multi-Layer Perceptrons (MLPs).

## Key Findings
- **High Redundancy**: Adjacent layers in a 10-layer MLP show >0.80 CKA similarity.
- **Width Matters**: Increasing layer width increases "transformation capacity," leading to lower similarity between layers.
- **Training Effect**: Trained networks have ~10% lower adjacent similarity than randomly initialized ones, as they learn to transform data into useful features.
- **Complexity**: More complex datasets (CIFAR-10) drive more significant representational changes across layers compared to simpler datasets (MNIST).

## Reproducing Results

1. **Setup Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn tqdm
   ```

2. **Run Experiments**:
   ```bash
   ./run_experiments.sh
   ```
   This script trains models on MNIST and CIFAR-10 and computes CKA matrices.

3. **View Analysis**:
   - Statistics: `python -m src.summary`
   - Visualizations: Check the `figures/` directory for heatmaps and line plots.

## Project Structure
- `src/model.py`: DeepMLP architecture with activation extraction.
- `src/train.py`: Training script.
- `src/analyze.py`: CKA computation and visualization.
- `src/summary.py`: Statistical summary of results.
- `REPORT.md`: Detailed research report and analysis.
