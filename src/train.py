import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import DeepMLP
import os
import json
import argparse
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    return train_loss / len(train_loader), 100. * correct / len(train_loader.dataset)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss / len(test_loader), 100. * correct / len(test_loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train Deep MLP on MNIST/CIFAR10')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--activation', type=str, default='relu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./datasets', train=False, transform=transform)
        input_size = 28 * 28
        output_size = 10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./datasets', train=False, transform=transform)
        input_size = 32 * 32 * 3
        output_size = 10

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DeepMLP(input_size, args.hidden_size, args.num_layers, output_size, args.activation).to(device)
    
    # Save random initialization
    model_name = f"{args.dataset}_h{args.hidden_size}_l{args.num_layers}_{args.activation}_s{args.seed}"
    random_path = os.path.join('results', f"{model_name}_random.pt")
    torch.save(model.state_dict(), random_path)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, args.epochs + 1):
        loss, acc = train(model, device, train_loader, optimizer, criterion)
        t_loss, t_acc = test(model, device, test_loader, criterion)
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Acc={acc:.2f}%, Test Loss={t_loss:.4f}, Acc={t_acc:.2f}%")
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['test_loss'].append(t_loss)
        history['test_acc'].append(t_acc)

    # Save trained model
    trained_path = os.path.join('results', f"{model_name}_trained.pt")
    torch.save(model.state_dict(), trained_path)
    
    with open(os.path.join('results', f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()
