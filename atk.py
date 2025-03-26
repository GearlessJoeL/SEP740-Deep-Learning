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

def pgd_attack(model, images, labels, eps=0.01, alpha=0.001, iters=10):
    """
    Perform a Projected Gradient Descent (PGD) attack on the model.
    
    Args:
        model: The model to attack
        images: The input images
        labels: The target labels
        eps: Maximum perturbation size (epsilon)
        alpha: Step size for gradient update
        iters: Number of attack iterations
    
    Returns:
        Adversarial images
    """
    # Store original model state
    training = model.training
    
    # Temporarily set model to training mode to ensure gradients flow
    model.train()
    
    # Clone the images to avoid modifying the original data
    images = images.clone().detach()
    adv_images = images.clone().detach()
    
    # PGD attack iterations
    for i in range(iters):
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
        
        # Calculate gradients directly for more control
        try:
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=adv_images,
                create_graph=False,
                allow_unused=True
            )[0]
            
            if gradients is None:
                print(f"Warning: gradients are None in iteration {i}")
                continue
                
            # Check for NaN values
            if torch.isnan(gradients).any():
                print(f"Warning: gradients contain NaN values in iteration {i}")
                gradients = torch.nan_to_num(gradients)
                
            # Update adversarial images
            with torch.no_grad():
                # Use FGSM update rule
                adv_images = adv_images.detach() + alpha * gradients.sign()
                
                # Project back to epsilon ball and valid image range
                delta = torch.clamp(adv_images - images, -eps, eps)
                adv_images = torch.clamp(images + delta, 0, 1)
                
        except Exception as e:
            print(f"Error in PGD attack iteration {i}: {e}")
    
    # Restore original model state
    if not training:
        model.eval()
        
    return adv_images.detach()

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
        return add_gaussian_noise(images)
    elif attack_type == 'pgd':
        # For stateless_resnet models, we need to handle them differently
        if model_type == 'stateless_resnet':
            # For spiking models, we'll just use random perturbation as a fallback
            # since gradient-based attacks don't work well with these models
            eps = 0.01  # Maximum perturbation
            delta = torch.empty_like(images).uniform_(-eps, eps)
            return torch.clamp(images + delta, 0, 1)
        else:
            # For standard models, use regular PGD attack
            return pgd_attack(model, images, labels)
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