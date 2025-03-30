import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_model
import numpy as np
from atk import add_gaussian_noise, pgd_attack, generate_adversarial_examples
from layers import *
import argparse

# Ensure weights directory exists
os.makedirs('./weight', exist_ok=True)

class SpikingMonitor:
    def __init__(self):
        self.spike_rates = {}
        self.membrane_potentials = {}
        self.thresh_crossings = {}

    def update(self, model, batch_idx):
        for name, module in model.named_modules():
            if isinstance(module, LIFSpike):
                if name not in self.spike_rates:
                    self.spike_rates[name] = []
                    self.membrane_potentials[name] = []
                    self.thresh_crossings[name] = []
                
                # Calculate spike rate
                if module.spike_count > 0:
                    spike_rate = module.spike_count / (module.T * module.mem.size(0))
                    self.spike_rates[name].append(spike_rate)
                
                # Track membrane potential statistics
                if module.mem is not None:
                    mem_mean = torch.mean(module.mem).item()
                    self.membrane_potentials[name].append(mem_mean)
                    
                    # Track threshold crossings
                    thresh_cross = torch.mean((module.mem > module.thresh).float()).item()
                    self.thresh_crossings[name].append(thresh_cross)

    def log_status(self, epoch, batch_idx):
        for name in self.spike_rates.keys():
            if len(self.spike_rates[name]) > 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Layer {name}:")
                print(f"  Spike Rate: {self.spike_rates[name][-1]:.4f}")
                print(f"  Membrane Potential: {self.membrane_potentials[name][-1]:.4f}")
                print(f"  Threshold Crossings: {self.thresh_crossings[name][-1]:.4f}")

def train(use_spike=False, atk='none', epochs=50, batch_size=32, lr=0.0001, model_type='standard', T=8, model_size=34):
    """
    Train the model and return detailed metrics
    
    Args:
        use_spike (bool): Whether to use spiking neurons
        atk (str): Attack type ('none', 'gn', or 'pgd')
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        model_type (str): Type of model to use ('standard', 'stateless_resnet')
        T (int): Number of time steps for spiking neurons
        model_size (int): Size of ResNet model (18 or 34)
    
    Returns:
        dict: Dictionary containing training metrics
    """
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use only 10% of the dataset
    train_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(test_dataset))
    train_indices = torch.randperm(len(dataset))[:train_size]
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Important: Set drop_last=True to ensure all batches have the same size
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize model - ensure all models use ResNet34 architecture
    model = get_model(
        num_classes=10, 
        use_spike=use_spike, 
        T=T, 
        model_type=model_type,
        model_size=model_size  # Explicitly set model size to 34
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if model_type == 'stateless_resnet':
        print(f"Training StatelessResNet{model_size} model with T={T} on {device}")
    else:
        print(f"Training {'spiking' if use_spike else 'non-spiking'} ResNet{model_size} model on {device}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Metrics storage
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    train_class_accuracies = []
    test_class_accuracies = []
    confusion_matrix = torch.zeros(10, 10)

    for epoch in range(epochs):
        # Reset membrane potentials
        if use_spike or model_type == 'stateless_resnet':
            model.apply(lambda m: m.reset_mem() if hasattr(m, 'reset_mem') else None)
            
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = torch.zeros(10, device=device)
        class_total = torch.zeros(10, device=device)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Apply attack if specified - use helper function
            if atk != 'none':
                try:
                    images = generate_adversarial_examples(
                        model=model,
                        images=images,
                        labels=labels,
                        attack_type=atk,
                        model_type=model_type
                    )
                except Exception as e:
                    print(f"Error generating adversarial examples: {e}")
                    # Continue with original images if attack fails

            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(images)
                
                # Handle spiking neuron output format (T, batch_size, num_classes)
                if (use_spike or model_type == 'stateless_resnet') and len(outputs.shape) == 3:
                    # Average over time dimension
                    outputs = outputs.mean(dim=0)
                
                # Ensure outputs and labels have the same batch size
                if outputs.size(0) != labels.size(0):
                    print(f"WARNING: Batch size mismatch - outputs: {outputs.size(0)}, labels: {labels.size(0)}")
                    min_batch = min(outputs.size(0), labels.size(0))
                    outputs = outputs[:min_batch]
                    labels = labels[:min_batch]
                
                # Calculate loss and backpropagate
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Per-class accuracy
                for i in range(10):
                    idx = (labels == i)
                    class_correct[i] += predicted[idx].eq(labels[idx]).sum().item()
                    class_total[i] += idx.sum().item()

                # Print batch progress
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                # Continue with the next batch
                continue
                    
            # Free memory every few batches
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

        # Update scheduler once per epoch
        scheduler.step(running_loss / len(train_loader))
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total if total > 0 else 0
        train_class_acc = 100. * class_correct / class_total.clamp(min=1)  # Avoid division by zero

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_class_accuracies.append(train_class_acc.cpu().numpy())

        # Testing phase
        model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(10, device=device)
        class_total = torch.zeros(10, device=device)
        confusion_matrix.zero_()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                try:
                    # Forward pass
                    outputs = model(images)
                    
                    # Handle spiking neuron output format (T, batch_size, num_classes)
                    if (use_spike or model_type == 'stateless_resnet') and len(outputs.shape) == 3:
                        # Average over time dimension
                        outputs = outputs.mean(dim=0)
                    
                    # Ensure outputs and labels have the same batch size
                    if outputs.size(0) != labels.size(0):
                        min_batch = min(outputs.size(0), labels.size(0))
                        outputs = outputs[:min_batch]
                        labels = labels[:min_batch]
                    
                    _, predicted = outputs.max(1)
                    
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # Per-class accuracy
                    for i in range(10):
                        idx = (labels == i)
                        class_correct[i] += predicted[idx].eq(labels[idx]).sum().item()
                        class_total[i] += idx.sum().item()

                    # Update confusion matrix
                    for t, p in zip(labels, predicted):
                        confusion_matrix[t.item(), p.item()] += 1
                except Exception as e:
                    print(f"Error in test batch: {e}")
                    continue

        # Calculate test metrics
        test_acc = 100. * correct / total if total > 0 else 0
        test_class_acc = 100. * class_correct / class_total.clamp(min=1)  # Avoid division by zero

        test_accuracies.append(test_acc)
        test_class_accuracies.append(test_class_acc.cpu().numpy())

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # Save the model
    model_name = f"{'stateless_resnet' if model_type == 'stateless_resnet' else 'resnet'}{model_size}_spike_{use_spike}_atk_{atk}.pth"
    torch.save(model.state_dict(), f"./weight/{model_name}")

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_class_accuracies': train_class_accuracies,
        'test_class_accuracies': test_class_accuracies,
        'final_confusion_matrix': confusion_matrix.cpu().numpy(),
        'model_path': f"./weight/{model_name}"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SNN models on MNIST.')
    parser.add_argument('--use_spike', action='store_true', help='Use spiking neurons (for standard ResNet)')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'stateless_resnet'],
                       help='Type of model to use (standard ResNet or StatelessResNet)')
    parser.add_argument('--model_size', type=int, default=34, choices=[18, 34],
                       help='Size of ResNet model (18 or 34)')
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'gn', 'pgd'],
                       help='Attack type (none, Gaussian noise, or PGD)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=8, help='Number of time steps for spiking networks')
    
    args = parser.parse_args()
    
    results = train(
        use_spike=args.use_spike,
        atk=args.attack,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        T=args.time_steps,
        model_size=args.model_size
    )
    
    print(f"Training completed. Final test accuracy: {results['test_accuracies'][-1]:.2f}%")
    print(f"Model saved to {results['model_path']}")