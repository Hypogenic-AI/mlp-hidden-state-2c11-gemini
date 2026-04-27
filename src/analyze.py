import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import DeepMLP
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def feature_space_linear_cka(features_x, features_y):
    """Computes Linear CKA in feature space."""
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    dot_product_similarity = torch.norm(torch.matmul(features_x.t(), features_y))**2
    normalization_x = torch.norm(torch.matmul(features_x.t(), features_x))
    normalization_y = torch.norm(torch.matmul(features_y.t(), features_y))
    
    return dot_product_similarity / (normalization_x * normalization_y)

def get_all_activations(model, device, loader, num_samples=1000):
    model.eval()
    all_activations = []
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            batch_activations = model.get_activations(data)
            
            if not all_activations:
                all_activations = [[] for _ in range(len(batch_activations))]
            
            for i, act in enumerate(batch_activations):
                all_activations[i].append(act)
            
            total_samples += data.size(0)
            if total_samples >= num_samples:
                break
                
    final_activations = []
    for i in range(len(all_activations)):
        concat_act = torch.cat(all_activations[i], dim=0)[:num_samples]
        final_activations.append(concat_act)
        
    return final_activations

def compute_cka_matrix(activations):
    num_layers = len(activations)
    cka_matrix = np.zeros((num_layers, num_layers))
    
    for i in tqdm(range(num_layers), desc="Computing CKA Matrix"):
        for j in range(i, num_layers):
            score = feature_space_linear_cka(activations[i], activations[j])
            cka_matrix[i, j] = score.item()
            cka_matrix[j, i] = score.item()
            
    return cka_matrix

def main():
    parser = argparse.ArgumentParser(description='Analyze MLP Representation Similarity')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./datasets', train=False, transform=transform)
        input_size = 28 * 28
        output_size = 10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_dataset = datasets.CIFAR10('./datasets', train=False, transform=transform)
        input_size = 32 * 32 * 3
        output_size = 10

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DeepMLP(input_size, args.hidden_size, args.num_layers, output_size, args.activation).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    activations = get_all_activations(model, device, test_loader, args.num_samples)
    cka_matrix = compute_cka_matrix(activations)
    
    base_name = os.path.basename(args.model_path).replace('.pt', '')
    np.save(os.path.join('results', f"{base_name}_cka.npy"), cka_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cka_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title(f"CKA Similarity Matrix: {base_name}")
    plt.savefig(os.path.join('figures', f"{base_name}_cka_heatmap.png"))
    plt.close()

    adj_cka = [cka_matrix[i, i+1] for i in range(len(cka_matrix)-1)]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cka_matrix)), adj_cka, marker='o')
    plt.title(f"Adjacent Layer Similarity: {base_name}")
    plt.grid(True)
    plt.savefig(os.path.join('figures', f"{base_name}_adjacent_cka.png"))
    plt.close()

if __name__ == '__main__':
    main()
