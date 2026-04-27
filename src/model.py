import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='relu'):
        super(DeepMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self._get_activation(activation))
            
        # Output layer
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def _get_activation(self, name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")
            
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_activations(self, x):
        """Returns activations after each linear layer."""
        x = x.view(x.size(0), -1)
        activations = []
        
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations.append(x.detach().cpu())
                
        return activations
