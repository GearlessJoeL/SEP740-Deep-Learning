import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_model, get_stateless_resnet34, get_resnet34
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from atk import add_gaussian_noise, pgd_attack, ATTACK_CONFIGS

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

def test(model_path, model_type='standard', batch_size=64):
    """
    Test the model with different attack intensities
    
    Args:
        model_path (str): Path to the .pth model file
        model_type (str): Type of model ('standard' or 'stateless_resnet')
        batch_size (int): Batch size for testing
    
    Returns:
        dict: Dictionary containing test results for different attack scenarios
    """
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
    
    if model_type == 'stateless_resnet':
        model = get_stateless_resnet34(num_classes=10, T=4)
    else:
        model = get_model(num_classes=10, model_type='standard')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define attack configurations
    attack_configs = {
        'clean': {'type': 'none', 'params': {}},
        'gaussian_weak': {'type': 'gn', 'params': {'std': 0.0001}},
        'gaussian_medium': {'type': 'gn', 'params': {'std': 0.001}},
        'gaussian_strong': {'type': 'gn', 'params': {'std': 0.01}},
        'pgd_weak': {'type': 'pgd', 'params': {'eps': 0.0001, 'alpha': 0.00001, 'iters': 10}},
        'pgd_medium': {'type': 'pgd', 'params': {'eps': 0.001, 'alpha': 0.0001, 'iters': 10}},
        'pgd_strong': {'type': 'pgd', 'params': {'eps': 0.01, 'alpha': 0.001, 'iters': 10}}
    }

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

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Apply attack
                if attack_config['type'] == 'gn':
                    images = add_gaussian_noise(images, **attack_config['params'])
                elif attack_config['type'] == 'pgd':
                    images = pgd_attack(model, images, labels, **attack_config['params'])

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
    resnet_model = get_stateless_resnet34(num_classes=10, T=4)
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