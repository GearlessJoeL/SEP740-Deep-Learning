import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_model, get_stateless_resnet34, get_resnet34, get_stateless_resnet18, get_resnet18
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Import attack functions - explicitly reimport to ensure we get the latest version
import importlib
import atk
importlib.reload(atk)
from atk import add_gaussian_noise, pgd_attack, ATTACK_CONFIGS

# Version with backward compatibility for use_spike parameter (2023-06-30)
# This version supports both model_type and the deprecated use_spike parameters

def evaluate_batch(model, images, labels, criterion, device):
    outputs = model(images)
    
    # Handle spiking neuron output format
    if len(outputs.shape) == 3:
        # Average over time dimension
        outputs = outputs.mean(dim=0)
    
    loss = criterion(outputs, labels)
    
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    
    # Per-class accuracy
    per_class_correct = torch.zeros(10, device=device)
    per_class_total = torch.zeros(10, device=device)
    for label, pred in zip(labels, predicted):
        per_class_correct[label] += (label == pred).item()
        per_class_total[label] += 1
    
    return loss.item(), correct, per_class_correct, per_class_total, predicted

def determine_model_architecture(model_path):
    """
    Analyze a model file to determine if it's ResNet18 or ResNet34
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        str: 'resnet18' or 'resnet34'
    """
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Count how many blocks in each layer
    layer_counts = {}
    for key in state_dict.keys():
        if key.startswith("layer"):
            parts = key.split(".")
            if len(parts) > 1:
                layer_name = parts[0]  # e.g., "layer1"
                block_num = int(parts[1])  # e.g., 0, 1, 2...
                
                if layer_name not in layer_counts or block_num > layer_counts[layer_name]:
                    layer_counts[layer_name] = block_num
    
    # Add 1 to get the number of blocks (since they're 0-indexed)
    for layer_name in layer_counts:
        layer_counts[layer_name] += 1
    
    # Extract the actual structure
    actual_structure = [
        layer_counts.get(f"layer{i+1}", 0) for i in range(4)
    ]
    
    # ResNet18: [2, 2, 2, 2], ResNet34: [3, 4, 6, 3]
    if actual_structure == [2, 2, 2, 2]:
        return 'resnet18'
    elif actual_structure == [3, 4, 6, 3]:
        return 'resnet34'
    else:
        # Default to ResNet18 for unknown structures
        print(f"Warning: Unknown model structure {actual_structure}, defaulting to ResNet18")
        return 'resnet18'

def test(model_path, model_type='standard', use_spike=False, batch_size=64, allow_non_strict=False, model_size=34):
    """
    Test the model with different attack intensities
    
    Args:
        model_path (str): Path to the .pth model file
        model_type (str): Type of model ('standard' or 'stateless_resnet')
        use_spike (bool): Deprecated parameter, use model_type='stateless_resnet' instead
        batch_size (int): Batch size for testing
        allow_non_strict (bool): Whether to allow non-strict model loading
        model_size (int): Size of ResNet model (18 or 34)
    
    Returns:
        dict: Dictionary containing test results for different attack scenarios
    """
    # Handle backward compatibility: if use_spike is True, override model_type
    if use_spike:
        model_type = 'stateless_resnet'
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Use 10% of test dataset
    test_size = int(0.1 * len(test_dataset))
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Initialize model based on model_type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the model based on model_type and model_size
    if model_type == 'stateless_resnet':
        if model_size == 18:
            model = get_stateless_resnet18(num_classes=10, T=4)
            print(f"Using StatelessResNet18 model with T=4")
        else:
            model = get_stateless_resnet34(num_classes=10, T=4)
            print(f"Using StatelessResNet34 model with T=4")
    else:
        if model_size == 18:
            model = get_resnet18(num_classes=10)
            print(f"Using standard ResNet18 model")
        else:
            model = get_resnet34(num_classes=10)
            print(f"Using standard ResNet34 model")
    
    # Load the model weights
    try:
        if allow_non_strict:
            print("Loading with strict=False to allow architecture mismatch")
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Attempting load with strict=False...")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Model loaded with missing keys ignored")
    
    model.to(device)
    
    # Don't set model.eval() here since each attack type will manage evaluation/training mode

    results = {}
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for attack_name, attack_config in ATTACK_CONFIGS.items():
        # Initialize metrics for current attack
        total_loss = 0
        correct = 0
        total = 0
        per_class_correct = torch.zeros(10, device=device)
        per_class_total = torch.zeros(10, device=device)
        all_preds = []
        all_targets = []

        # Set model to appropriate mode for each attack type
        if attack_config['type'] == 'pgd':
            # For PGD attack, we'll set the model to eval mode in the attack function
            # but will make sure gradients are computed
            model.eval()
            # Enable gradients for model parameters during attack
            for param in model.parameters():
                param.requires_grad = True
        else:
            # For clean and Gaussian noise evaluation, no gradients are needed
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply attack
            if attack_config['type'] == 'gn':
                images = add_gaussian_noise(images, **attack_config['params'])
            elif attack_config['type'] == 'pgd':
                # Pass model_type to the pgd_attack function
                params = {k: v for k, v in attack_config['params'].items()}
                images = pgd_attack(
                    model=model, 
                    images=images, 
                    labels=labels, 
                    eps=params.get('eps', 0.01),
                    alpha=params.get('alpha', 0.001),
                    iters=params.get('iters', 10),
                    model_type=model_type
                )

            # For evaluation, we always want no_grad
            with torch.no_grad():
                # Evaluate batch
                batch_loss, batch_correct, batch_class_correct, batch_class_total, predicted = \
                    evaluate_batch(model, images, labels, criterion, device)
                
                total_loss += batch_loss
                correct += batch_correct
                total += labels.size(0)
                per_class_correct += batch_class_correct
                per_class_total += batch_class_total
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Calculate metrics
        test_loss = total_loss / total
        test_acc = 100. * correct / total
        class_accuracies = 100. * per_class_correct / per_class_total
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # Find most confused pairs
        confused_pairs = [(i, j, conf_matrix[i, j]) 
                         for i in range(10) for j in range(10) if i != j]
        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        results[attack_name] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'class_accuracies': class_accuracies.cpu().tolist(),
            'confusion_matrix': conf_matrix,
            'confused_pairs': confused_pairs[:5],
            'per_class_correct': per_class_correct.cpu().tolist(),
            'per_class_total': per_class_total.cpu().tolist()
        }

    # Make sure model is in eval mode when returning
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return results

def test_models():
    # Create random input data (MNIST shape)
    batch_size = 2
    x = torch.randn(batch_size, 1, 28, 28)
    
    # Test ResNet34
    print("Testing ResNet34...")
    resnet34_model = get_model(num_classes=10, model_type='standard')
    resnet34_model.eval()
    
    with torch.no_grad():
        resnet34_output = resnet34_model(x)
        print(f"ResNet34 output shape: {resnet34_output.shape}")
    
    # Test StatelessResNet
    print("\nTesting StatelessResNet...")
    resnet_model = get_model(num_classes=10, model_type='stateless_resnet', T=4)
    resnet_model.eval()
    
    with torch.no_grad():
        resnet_output = resnet_model(x)
        print(f"StatelessResNet output shape: {resnet_output.shape}")
    
    print("\nBoth models successfully imported and tested!")

if __name__ == "__main__":
    weight_dir = "./weight"
    for model_file in os.listdir(weight_dir):
        if model_file.endswith(".pth"):
            print(f"\nTesting model: {model_file}")
            
            # Determine model type from filename
            model_type = 'stateless_resnet' if 'stateless_resnet' in model_file else 'standard'
            
            results = test(
                model_path=os.path.join(weight_dir, model_file),
                model_type=model_type
            )
            
            for attack_name, attack_results in results.items():
                print(f"\nAttack scenario: {attack_name}")
                print(f"Test Accuracy: {attack_results['test_acc']:.2f}%")
                print("Per-class accuracies:")
                for i, acc in enumerate(attack_results['class_accuracies']):
                    print(f"Digit {i}: {acc:.2f}%")
                print("\nTop 5 most confused pairs (true_label, predicted_label, count):")
                for pair in attack_results['confused_pairs']:
                    print(f"  {pair[0]} → {pair[1]}: {pair[2]}")

    test_models()