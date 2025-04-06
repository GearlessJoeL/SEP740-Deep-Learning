import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from model import get_model
import argparse
from atk import generate_adversarial_examples

# Ensure weights directory exists
os.makedirs('./weight', exist_ok=True)

class MixedDataset(Dataset):
    """Dataset that mixes raw and normalized images based on a proportion"""
    def __init__(self, dataset, raw_proportion=0.0):
        self.dataset = dataset
        self.raw_proportion = raw_proportion
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # If random number is less than raw_proportion, return raw image
        if torch.rand(1).item() < self.raw_proportion:
            # Skip normalization for this image
            return image, label
        else:
            # Apply normalization
            return self.normalize(image), label

def train(use_spike=False, atk='none', epochs=50, batch_size=64, lr=0.001, 
          model_type='standard', T=4, optimizer_type='adamw', raw_prop=0.0):
    """
    Train the model using the original procedure with just 10% of the data and adversarial attacks
    Now supporting different optimizers: SGD, Adam, and AdamW
    raw_prop: proportion of raw (unnormalized) images to use in training (0.0 to 1.0)
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalization will be applied selectively in the MixedDataset
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use 10% of data for faster training
    train_size = int(0.1 * len(train_dataset))
    test_size = int(0.1 * len(test_dataset))
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Wrap datasets with MixedDataset to control raw image proportion
    train_mixed = MixedDataset(train_subset, raw_prop)
    test_mixed = MixedDataset(test_subset, raw_prop)

    train_loader = DataLoader(train_mixed, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_mixed, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Training on {train_size} samples, testing on {test_size} samples")
    print(f"Attack type: {atk}, Optimizer: {optimizer_type}, Raw image proportion: {raw_prop}")

    # Initialize model
    model = get_model(
        num_classes=10, 
        use_spike=use_spike, 
        T=T, 
        model_type=model_type
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Configure optimizer based on optimizer_type
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    else:  # default to adamw
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    
    # Use OneCycleLR scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr*10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Warm up quickly
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Apply attack if specified (to 70% of the batch)
            if atk != 'none':
                try:
                    # Determine batch size and how many images to attack (70%)
                    batch_size = images.size(0)
                    attack_size = int(batch_size * 0.7)
                    
                    if attack_size > 0:
                        # Randomly select which images to attack
                        indices = torch.randperm(batch_size, device=device)
                        attack_indices = indices[:attack_size]
                        
                        # Create copies of the images to be attacked
                        images_to_attack = images[attack_indices].clone()
                        images_to_attack.requires_grad_(True)
                        
                        # Generate adversarial examples
                        adv_images = generate_adversarial_examples(
                            model=model,
                            images=images_to_attack,
                            labels=labels[attack_indices],
                            attack_type=atk,
                            eps=0.0001
                        )
                        
                        # Replace only the selected images with their adversarial versions
                        images[attack_indices] = adv_images.detach()
                except Exception as e:
                    print(f"Attack failed: {e}")
                    # Continue with original images if attack fails
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle spiking model outputs
            if len(outputs.shape) == 3:  # [time_steps, batch_size, num_classes]
                outputs = outputs.mean(dim=0)  # Average over time dimension
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # if (i + 1) % 10 == 0:
            #     print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, '
            #           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, '
            #           f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Also test against the attack
                if atk != 'none':
                    try:
                        images.requires_grad_(True)
                        adv_images = generate_adversarial_examples(
                            model=model,
                            images=images,
                            labels=labels,
                            attack_type=atk,
                            eps=0.0001
                        )
                        images = adv_images.detach() 
                    except Exception as e:
                        print(f"Error generating adversarial examples during testing: {e}")
                
                outputs = model(images)
                
                if len(outputs.shape) == 3:
                    outputs = outputs.mean(dim=0)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_name = f"{model_type}_spike_{use_spike}_atk_{atk}_{optimizer_type}_raw_prop_{raw_prop:.2f}.pth"
            torch.save(model.state_dict(), f"./weight/{model_name}")
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        scheduler.step()
    
    print(f'Training completed for {model_name}')
    print(f'Final test accuracy: {best_acc:.2f}%')
    print(f'Model saved to: ./weight/{model_name}')
    
    return {
        'best_acc': best_acc,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'raw_prop': raw_prop
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SNN models on MNIST.')
    parser.add_argument('--use_spike', action='store_true', help='Use spiking neurons')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard'],
                       help='Type of model to use')
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'gn', 'pgd'],
                       help='Attack type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=4, help='Number of time steps for spiking networks')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer to use for training')
    parser.add_argument('--raw_prop', type=float, default=None, 
                       help='Proportion of raw images to use (0.0-1.0). If not specified, will train with all proportions [0, 0.25, 0.5, 0.75]')
    
    args = parser.parse_args()
    
    # If raw_prop is provided, train with just that proportion
    if args.raw_prop is not None:
        train_history = train(
            use_spike=args.use_spike,
            atk=args.attack,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_type=args.model_type,
            T=args.time_steps,
            optimizer_type=args.optimizer,
            raw_prop=args.raw_prop
        )
        print(f"Training completed with raw_prop={args.raw_prop}. Best test accuracy: {train_history['best_acc']:.2f}%")
    else:
        # Train with different proportions as specified
        proportions = [0.0, 0.25, 0.5, 0.75]
        results = []
        
        for prop in proportions:
            print(f"\n\n========== Training with raw image proportion: {prop} ==========\n")
            train_history = train(
                use_spike=args.use_spike,
                atk=args.attack,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                model_type=args.model_type,
                T=args.time_steps,
                optimizer_type=args.optimizer,
                raw_prop=prop
            )
            results.append(train_history)
            print(f"Training completed with raw_prop={prop}. Best test accuracy: {train_history['best_acc']:.2f}%")
        
        # Print summary of results
        print("\n\n========== Summary of Results ==========")
        for idx, res in enumerate(results):
            print(f"Raw Proportion {proportions[idx]:.2f}: Test Accuracy {res['best_acc']:.2f}%")