import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import importlib
import sys

# Reimport modules
from model import get_model
from layers import LIFSpike  # Update to correct class name
import atk
from atk import add_gaussian_noise, pgd_attack, ATTACK_CONFIGS, generate_adversarial_examples

# Version updated to work with the new WideResNet model (2024-06-01)

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

def test(model, attack_type='none', batch_size=256, device='cuda', eps=0.1, proportion=0.1, optimizer='adamw'):
    """
    Test the model with specified attack type
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10  # For MNIST, 10 classes
    class_total = [0] * 10
    
    # Initialize confusion matrix
    conf_matrix = torch.zeros(10, 10, device=device)
    
    # Load test data with larger batch size
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    
    dataset_size = len(test_dataset)
    subset_size = int(dataset_size * proportion)
    indices = torch.randperm(dataset_size)[:subset_size]
    test_subset = Subset(test_dataset, indices)
    
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # print(f"Testing on {subset_size} samples ({proportion*100:.1f}% of test set)")
    print(f"In test function, attack_type: {attack_type}, optimizer: {optimizer}")
    
    # Check if model is spiking or non-spiking
    is_spiking = hasattr(model, 'T') and model.T > 0
    print(f"Model is {'spiking' if is_spiking else 'non-spiking'}")
    
    # Debug: Print model parameters to check if they're loaded correctly
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")
    
    # Save sample images for debugging
    debug_images = []
    debug_outputs = []
    debug_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Save original images for debugging (first batch only)
            if batch_idx == 0 and len(debug_images) < 5:
                debug_images.append(("original", images[:5].clone()))
            
            # Apply attack if specified
            if attack_type != 'none':
                try:
                    # Enable gradients for input images
                    images.requires_grad = True
                    
                    # Generate adversarial examples
                    adv_images = generate_adversarial_examples(
                        model=model,
                        images=images,
                        labels=labels,
                        attack_type=attack_type,
                        eps=eps
                    )
                    
                    # Save adversarial images for debugging (first batch only)
                    if batch_idx == 0 and len(debug_images) < 10:
                        debug_images.append((f"{attack_type}_attack", adv_images[:5].clone()))
                    
                    # Disable gradients after attack
                    images = adv_images.detach()
                except Exception as e:
                    print(f"Warning: Attack generation failed, using original images. Error: {e}")
                    images = images.detach()
            
            # Forward pass
            outputs = model(images)
            
            # Save outputs for debugging (first batch only)
            if batch_idx == 0 and len(debug_outputs) < 5:
                debug_outputs.append((attack_type, outputs[:5].clone()))
                debug_labels.append(labels[:5].clone())
            
            # Handle spiking neuron outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Average over time dimension if needed
            if len(outputs.shape) > 2:
                outputs = outputs.mean(dim=0)
            
            # Get predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels, predicted):
                conf_matrix[t.long(), p.long()] += 1
            
            # Calculate per-class accuracy
            c = predicted.eq(labels)
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Print debug information
    print("\nDEBUG INFORMATION:")
    print("Sample predictions:")
    for i, ((attack_name, outputs), labels) in enumerate(zip(debug_outputs, debug_labels)):
        print(f"\nSample outputs for {attack_name}:")
        for j in range(min(5, outputs.size(0))):
            if len(outputs.shape) > 2:  # Handle spiking outputs
                output = outputs[:, j].mean(dim=0)
            else:
                output = outputs[j]
            
            _, pred = output.max(0)
            print(f"  Sample {j}: Label={labels[j].item()}, Predicted={pred.item()}")
            print(f"  Logits: {output.cpu().numpy()}")
    
    # Calculate overall accuracy
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        print(f"  Class {i}: {class_accuracies[i]:.2f}%")
    
    # Convert confusion matrix to numpy and move to CPU
    conf_matrix = conf_matrix.cpu().numpy()
    
    return accuracy, class_accuracies, conf_matrix

def test_all_models(attack_types=['none', 'gn', 'pgd'], batch_size=256, device='cuda', eps=0.1, proportion=0.1):
    """
    Test all models in the weight directory with specified attack types
    """
    results = {}
    
    # Get all model files
    model_files = [f for f in os.listdir('weight') if f.endswith('.pth')]
    print(f"Found {len(model_files)} model files: {model_files}")
    
    for model_file in model_files:
        print(f"\nTesting {model_file}...")
        
        # Extract model parameters from filename
        params = model_file.replace('.pth', '').split('_')
        
        # Handle different filename formats
        if len(params) >= 4:  # Updated to account for optimizer in filename
            if params[0] == 'standard':
                model_type = 'standard'  # Always use 'standard' for get_model function
                use_spike = params[2] == 'True'  # Convert to boolean
                model_size = 16  # Default size for WideResNet
                optimizer = params[3] if len(params) >= 4 else 'adamw'  # Extract optimizer
                print(f"Detected model parameters: model_type={model_type}, use_spike={use_spike}, model_size={model_size}, optimizer={optimizer}")
            else:
                print(f"Warning: Unexpected filename format for {model_file}, skipping...")
                continue
        else:
            print(f"Warning: Unexpected filename format for {model_file}, skipping...")
            continue
        
        # Create model with DEBUG output
        try:
            print(f"Creating model with params: model_type={model_type}, use_spike={use_spike}, model_size={model_size}")
            model = get_model(
                num_classes=10,
                use_spike=use_spike,
                T=4,  # Default timesteps for spiking models
                model_type=model_type,
                model_size=model_size
            )
            print(f"Model created successfully: {type(model).__name__}")
        except Exception as e:
            print(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Load weights
        try:
            state_dict = torch.load(os.path.join('weight', model_file))
            print(f"State dict loaded with {len(state_dict)} keys")
            
            # Debug: print model state_dict keys vs loaded keys
            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            if model_keys != loaded_keys:
                missing_in_model = loaded_keys - model_keys
                missing_in_loaded = model_keys - loaded_keys
                
                if missing_in_model:
                    print(f"Keys in loaded state_dict but not in model: {missing_in_model}")
                if missing_in_loaded:
                    print(f"Keys in model but not in loaded state_dict: {missing_in_loaded}")
            
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            print(f"Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test with each attack type
        model_results = {}
        for attack in attack_types:
            print(f"  Testing with {attack} attack...")
            try:
                acc, class_accs, conf_matrix = test(
                    model=model,
                    attack_type=attack,
                    batch_size=batch_size,
                    device=device,
                    eps=eps,
                    proportion=proportion,
                    optimizer=optimizer.lower()  # Pass optimizer to test function
                )
                
                model_results[attack] = {
                    'accuracy': acc,
                    'class_accuracies': class_accs,
                    'confusion_matrix': conf_matrix
                }
                
                print(f"    Accuracy: {acc:.2f}%")
            except Exception as e:
                print(f"    Error during testing: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if model_results:  # Only add if we have results
            results[model_file] = model_results
    
    print("\nFinal results structure:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for attack_name, attack_results in model_results.items():
            print(f"  {attack_name}: {attack_results['accuracy']:.2f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SNN models on MNIST.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--use_spike', action='store_true', help='Use spiking neurons')
    parser.add_argument('--attack_type', type=str, default='none', choices=['none', 'gn', 'pgd'],
                      help='Attack type')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard'],
                      help='Type of model to use')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--time_steps', type=int, default=4, help='Number of time steps for spiking networks')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for adversarial attack')
    parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of test set to use (0-1)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],
                      help='Optimizer used for training')
    args = parser.parse_args()
    
    # Test the test function directly
    print("Testing the test function directly...")
    
    # Create a model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(
        num_classes=10,
        use_spike=args.use_spike,
        T=args.time_steps,
        model_type=args.model_type
    )
    
    # If model path is provided, load weights
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Using model with random weights (no model path provided)")
    
    model = model.to(device)
    model.eval()
    
    # Run the test function
    print(f"Running test with attack_type={args.attack_type}, optimizer={args.optimizer}")
    acc, per_class_acc, conf_matrix = test(
        model=model,
        attack_type=args.attack_type,
        batch_size=args.batch_size,
        device=device,
        eps=args.eps,
        proportion=args.proportion,
        optimizer=args.optimizer.lower()
    )
    
    print(f"\nTest Results:")
    print(f"Overall accuracy: {acc:.2f}%")
    print("\nPer-class accuracy summary:")
    for i, class_acc in enumerate(per_class_acc):
        print(f"  Class {i}: {class_acc:.2f}%")