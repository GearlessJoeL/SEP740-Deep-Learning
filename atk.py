import torch
import torch.nn as nn
import torch.nn.functional as F

def add_gaussian_noise(images, std=0.0001):
    """Add Gaussian noise to images.
    
    Args:
        images (torch.Tensor): Input images
        std (float): Standard deviation of the Gaussian noise
    
    Returns:
        torch.Tensor: Noisy images
    """
    noise = torch.randn_like(images) * std
    return images + noise

def pgd_attack(model, images, labels, eps=0.01, alpha=0.001, iters=10, model_type='standard'):
    """
    Perform a Projected Gradient Descent (PGD) attack on the model.
    
    Args:
        model: The model to attack
        images: The input images
        labels: The target labels
        eps: Maximum perturbation size (epsilon)
        alpha: Step size for gradient update
        iters: Number of attack iterations
        model_type: Type of model ('standard' or 'stateless_resnet')
    
    Returns:
        Adversarial images
    """
    # If it's a spiking model, use a simplified attack
    if model_type == 'stateless_resnet' or hasattr(model, 'T'):
        return spiking_pgd_attack(model, images, labels, eps, alpha, iters)
    
    # Store original model state
    training = model.training
    
    # Set model to evaluation mode, but make sure requires_grad is enabled for relevant parameters
    model.eval()
    
    # Ensure model parameters have requires_grad enabled during the attack
    for param in model.parameters():
        param.requires_grad = True
    
    # Clone the images to avoid modifying the original data
    images = images.clone().detach()
    adv_images = images.clone().detach()
    
    # PGD attack iterations
    for i in range(iters):
        try:
            # Setup for gradient calculation
            adv_images = adv_images.detach().requires_grad_(True)
            
            # Forward pass
            outputs = model(adv_images)
            
            # Handle spiking model output format
            if len(outputs.shape) == 3:  # Shape: [time_steps, batch_size, num_classes]
                outputs = outputs.mean(dim=0)  # Average over time dimension
            
            # Calculate loss (maximizing the original class loss)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            
            # Make sure loss requires grad
            if not loss.requires_grad:
                print(f"Warning: loss doesn't require grad in iteration {i}")
                continue
                
            # Calculate gradients
            loss.backward()
            
            # Get gradients from adv_images
            if adv_images.grad is None:
                print(f"Warning: gradients are None in iteration {i}")
                continue
                
            # Update adversarial images
            with torch.no_grad():
                # Use FGSM update rule
                adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
                
                # Project back to epsilon ball and valid image range
                delta = torch.clamp(adv_images - images, -eps, eps)
                adv_images = torch.clamp(images + delta, 0, 1)
                
        except Exception as e:
            print(f"Error in PGD attack iteration {i}: {e}")
    
    # Restore original model state
    if not training:
        model.eval()
        # Reset requires_grad to False for evaluation
        for param in model.parameters():
            param.requires_grad = False
        
    return adv_images.detach()

def spiking_pgd_attack(model, images, labels, eps=0.01, alpha=0.001, iters=10):
    """
    Perform a PGD attack specifically designed for spiking neural networks.
    Uses surrogate gradients and random search when gradients are not available.
    
    Args:
        model: The spiking neural network model
        images: The input images
        labels: The target labels
        eps: Maximum perturbation size (epsilon)
        alpha: Step size for gradient update
        iters: Number of attack iterations
        
    Returns:
        Adversarial images
    """
    # Clone the images to avoid modifying the original data
    images = images.clone().detach()
    adv_images = images.clone().detach()
    
    # Check model mode and temporarily set to evaluation
    training = model.training
    model.eval()
    
    # Get initial predictions
    with torch.no_grad():
        initial_output = model(images)
        if len(initial_output.shape) == 3:  # Spiking output [time_steps, batch_size, num_classes]
            initial_output = initial_output.mean(dim=0)
        _, initial_pred = initial_output.max(1)
    
    # Best adversarial examples so far
    best_adv = adv_images.clone()
    
    # PGD attack iterations
    for i in range(iters):
        # Try using gradients first
        try:
            adv_images.requires_grad_(True)
            
            # Forward pass
            outputs = model(adv_images)
            
            # Average over time steps if spiking model
            if len(outputs.shape) == 3:
                outputs = outputs.mean(dim=0)
            
            # Calculate loss (cross-entropy)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            
            # Calculate gradients
            loss.backward()
            
            # Check if we have gradients
            if adv_images.grad is not None:
                # Update adversarial images with sign of gradient
                with torch.no_grad():
                    adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
                    delta = torch.clamp(adv_images - images, -eps, eps)
                    adv_images = torch.clamp(images + delta, 0, 1)
            else:
                # If no gradients, use random search
                print(f"Using random search in iteration {i} (no gradients)")
                directions = torch.randn_like(adv_images).sign() * alpha
                candidates = torch.clamp(adv_images + directions, 0, 1)
                delta = torch.clamp(candidates - images, -eps, eps)
                adv_images = torch.clamp(images + delta, 0, 1)
        
        except Exception as e:
            # If error occurs, use random search
            print(f"Using random search in iteration {i} ({e})")
            directions = torch.randn_like(adv_images).sign() * alpha
            candidates = torch.clamp(adv_images + directions, 0, 1)
            delta = torch.clamp(candidates - images, -eps, eps)
            adv_images = torch.clamp(images + delta, 0, 1)
        
        # Check if current adversarial examples are better than previous best
        with torch.no_grad():
            outputs = model(adv_images)
            if len(outputs.shape) == 3:
                outputs = outputs.mean(dim=0)
            _, adv_pred = outputs.max(1)
            
            # Update best adversarial examples where the attack is successful
            is_successful = (adv_pred != labels) & (initial_pred == labels)
            best_adv[is_successful] = adv_images[is_successful]
    
    # Restore original model state
    if training:
        model.train()
    
    return best_adv.detach()

def generate_adversarial_examples(model, images, labels, attack_type='none', model_type='standard'):
    """
    Generate adversarial examples for training or testing.
    
    Args:
        model: The model to attack
        images: The input images
        labels: The target labels
        attack_type: Type of attack ('none', 'gn', or 'pgd')
        model_type: Type of model ('standard' or 'stateless_resnet')
    
    Returns:
        Adversarial images
    """
    if attack_type == 'none':
        return images
    elif attack_type == 'gn':
        return add_gaussian_noise(images, **ATTACK_CONFIGS['gaussian_medium']['params'])
    elif attack_type == 'pgd':
        # Extract parameters from attack config
        params = ATTACK_CONFIGS['pgd_medium']['params']
        # Call pgd_attack with explicit named parameters
        return pgd_attack(
            model=model, 
            images=images, 
            labels=labels, 
            eps=params.get('eps', 0.01),
            alpha=params.get('alpha', 0.001),
            iters=params.get('iters', 10),
            model_type=model_type
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

# Define standard attack configurations
ATTACK_CONFIGS = {
    'clean': {'type': 'none', 'params': {}},
    'gaussian_weak': {'type': 'gn', 'params': {'std': 0.0001}},
    'gaussian_medium': {'type': 'gn', 'params': {'std': 0.001}},
    'gaussian_strong': {'type': 'gn', 'params': {'std': 0.01}},
    'pgd_weak': {'type': 'pgd', 'params': {'eps': 0.0001, 'alpha': 0.00001, 'iters': 10}},
    'pgd_medium': {'type': 'pgd', 'params': {'eps': 0.001, 'alpha': 0.0001, 'iters': 10}},
    'pgd_strong': {'type': 'pgd', 'params': {'eps': 0.01, 'alpha': 0.001, 'iters': 10}}
}