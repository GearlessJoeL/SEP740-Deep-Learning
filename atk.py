import torch
import torch.nn as nn
import torch.nn.functional as F

def add_gaussian_noise(images, std=0.1):
    """Add Gaussian noise to images"""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)

def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=10):
    """Improved PGD attack implementation"""
    # Save original model state
    was_training = model.training
    model.eval()  # Set to eval mode but enable gradients
    
    # Initialize adversarial examples
    orig_images = images.clone().detach()
    adv_images = orig_images + torch.empty_like(orig_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1).requires_grad_(True)
    
    for _ in range(iters):
        with torch.enable_grad():
            # Forward pass
            outputs = model(adv_images)
            
            # Handle spiking neuron output format
            if len(outputs.shape) == 3:
                outputs = outputs.mean(dim=0)
            
            # Calculate loss (we want to maximize this loss)
            loss = F.cross_entropy(outputs, labels)
        
        # Calculate gradient
        grad = torch.autograd.grad(loss, adv_images, only_inputs=True)[0]
        
        # Update adversarial examples (maximize loss by following gradient)
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball and valid image range
        delta = torch.clamp(adv_images - orig_images, -eps, eps)
        adv_images = torch.clamp(orig_images + delta, 0, 1).requires_grad_(True)
    
    # Restore original model state
    if was_training:
        model.train()
    else:
        model.eval()
    
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

def generate_adversarial_examples(model, images, labels, attack_type, eps=0.1):
    """
    Generate adversarial examples using specified attack type
    
    Args:
        model: The model to attack
        images: Input images
        labels: Target labels
        attack_type: Type of attack ('gn' or 'pgd')
        eps: Maximum perturbation size
    """
    if attack_type == 'gn':
        # Gaussian noise attack
        return add_gaussian_noise(images, eps)
    
    elif attack_type == 'pgd':
        # Use PGD attack
        return pgd_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=eps/4,  # Step size
            iters=10      # Number of iterations
        )
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

# Attack configurations
ATTACK_CONFIGS = {
    'gn': {'eps': 0.1},
    'pgd': {'eps': 0.1, 'alpha': 0.01, 'iters': 10}
}