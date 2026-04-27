import torch

def feature_space_linear_cka(features_x, features_y):
    """
    Computes Linear CKA in feature space (efficient for N > D).
    
    Args:
        features_x: (N, D1) matrix of features.
        features_y: (N, D2) matrix of features.
    
    Returns:
        The CKA similarity score (scalar).
    """
    # Centering features
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    # HSIC (Hilbert-Schmidt Independence Criterion) in feature space
    dot_product_similarity = torch.norm(torch.matmul(features_x.t(), features_y))**2
    
    # Normalization terms
    normalization_x = torch.norm(torch.matmul(features_x.t(), features_x))
    normalization_y = torch.norm(torch.matmul(features_y.t(), features_y))
    
    return dot_product_similarity / (normalization_x * normalization_y)

def gram_linear_cka(gram_x, gram_y):
    """
    Computes Linear CKA using Gram matrices (efficient for D > N).
    
    Args:
        gram_x: (N, N) Gram matrix (X @ X.T).
        gram_y: (N, N) Gram matrix (Y @ Y.T).
        
    Returns:
        The CKA similarity score (scalar).
    """
    # Centering Gram matrices
    n = gram_x.size(0)
    unit = torch.ones([n, n], device=gram_x.device)
    identity = torch.eye(n, device=gram_x.device)
    centering_matrix = identity - (unit / n)
    
    gram_x_centered = centering_matrix @ gram_x @ centering_matrix
    gram_y_centered = centering_matrix @ gram_y @ centering_matrix
    
    # HSIC
    hsic = (gram_x_centered * gram_y_centered).sum()
    
    # Normalization
    normalization_x = torch.norm(gram_x_centered)
    normalization_y = torch.norm(gram_y_centered)
    
    return hsic / (normalization_x * normalization_y)

if __name__ == "__main__":
    # Example usage
    X = torch.randn(100, 64)
    Y = X @ torch.randn(64, 64) + 0.1 * torch.randn(100, 64)
    
    score = feature_space_linear_cka(X, Y)
    print(f"CKA Score (Feature Space): {score.item():.4f}")
    
    G_X = X @ X.t()
    G_Y = Y @ Y.t()
    score_gram = gram_linear_cka(G_X, G_Y)
    print(f"CKA Score (Gram Space): {score_gram.item():.4f}")
