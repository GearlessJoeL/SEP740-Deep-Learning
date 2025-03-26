import torch
from layers import *

import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Normalization for MNIST
        self.norm = TensorNormalization((0.1307,), (0.3081,))
        
        # Initial layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # Normalize input
        x = self.norm(x)
        
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and final classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class StatelessResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, T=4):
        super(StatelessResNetBlock, self).__init__()
        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Spiking parameters    
        self.thresh = 1.0
        self.tau = 0.5
    
    def forward(self, x, mem1=None, mem2=None):
        # Compute shortcut
        identity = self.shortcut(x)
        
        # First convolution
        conv1_out = self.bn1(self.conv1(x))
        
        # Initialize membrane potentials if not provided
        if mem1 is None:
            mem1 = torch.zeros_like(conv1_out)
        
        # Update membrane potential and compute spike
        mem1 = mem1 * self.tau + conv1_out * (1 - self.tau)
        spike1 = (mem1 >= self.thresh).float()
        mem1 = mem1 * (1 - spike1)  # Reset membrane potential
        
        # Second convolution
        conv2_out = self.bn2(self.conv2(spike1))
        
        # Initialize second membrane potential if not provided
        if mem2 is None:
            mem2 = torch.zeros_like(conv2_out)
        
        # Update membrane potential and compute spike
        mem2 = mem2 * self.tau + conv2_out * (1 - self.tau)
        spike2 = (mem2 >= self.thresh).float()
        mem2 = mem2 * (1 - spike2)  # Reset membrane potential
        
        # Ensure identity and spike2 have exactly the same dimensions
        if identity.shape != spike2.shape:
            # Resize identity to match spike2's dimensions
            identity = F.interpolate(identity, size=(spike2.size(2), spike2.size(3)), mode='nearest')
        
        # Residual connection (average with identity to avoid in-place operations)
        out = 0.5 * (spike2 + identity)
        
        return out, mem1, mem2

class StatelessResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, T=4):
        super(StatelessResNet, self).__init__()
        self.T = T
        self.in_channels = 64
        
        # Normalization for MNIST
        self.norm = TensorNormalization((0.1307,), (0.3081,))
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Spiking parameters
        self.thresh = 1.0
        self.tau = 0.5
        
        # ResNet layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Final classifier
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(StatelessResNetBlock(self.in_channels, out_channels, stride, self.T))
            self.in_channels = out_channels
        return nn.ModuleList(layers)
    
    def forward(self, x):
        # Normalize input
        x = self.norm(x)
        batch_size = x.size(0)
        device = x.device
        
        # Initial convolution - do this outside the loop and reuse
        conv1_out = self.bn1(self.conv1(x))
        
        # Initialize the first membrane potential
        mem_conv1 = torch.zeros_like(conv1_out)
        
        # Simulate over T time steps
        outputs = []
        for t in range(self.T):
            # Process the first layer at each time step
            mem_conv1 = mem_conv1 * self.tau + conv1_out * (1 - self.tau)
            spike = (mem_conv1 >= self.thresh).float()
            mem_conv1 = mem_conv1 * (1 - spike)
            
            # Store memory states for all block layers
            curr_spike = spike
            mems1 = []
            mems2 = []
            
            # Process through ResNet blocks with fresh spikes each time step
            # Layer 1
            for j, block in enumerate(self.layer1):
                curr_spike, mem1, mem2 = block(curr_spike, None, None)
                mems1.append(mem1)
                mems2.append(mem2)
            
            # Layer 2
            for j, block in enumerate(self.layer2):
                curr_spike, mem1, mem2 = block(curr_spike, None, None)
                mems1.append(mem1)
                mems2.append(mem2)
            
            # Layer 3
            for j, block in enumerate(self.layer3):
                curr_spike, mem1, mem2 = block(curr_spike, None, None)
                mems1.append(mem1)
                mems2.append(mem2)
            
            # Layer 4
            for j, block in enumerate(self.layer4):
                curr_spike, mem1, mem2 = block(curr_spike, None, None)
                mems1.append(mem1)
                mems2.append(mem2)
            
            # Global average pooling
            pooled = F.adaptive_avg_pool2d(curr_spike, 1).view(batch_size, -1)
            
            # Final classification
            out = self.fc(pooled)
            outputs.append(out)
        
        # Stack outputs over time
        return torch.stack(outputs)
    
    def reset_mem(self):
        """Reset all membrane potentials in the network"""
        # No persistent state to reset in this implementation
        pass

def get_stateless_resnet18(num_classes=10, T=4):
    """Returns a stateless ResNet-18 model designed for spiking neurons."""
    return StatelessResNet([2, 2, 2, 2], num_classes, T)

def get_stateless_resnet34(num_classes=10, T=4):
    """Returns a stateless ResNet-34 model designed for spiking neurons."""
    return StatelessResNet([3, 4, 6, 3], num_classes, T)

def get_resnet34(num_classes=10):
    """Returns a standard ResNet-34 model."""
    return ResNet(num_blocks=[3, 4, 6, 3], num_classes=num_classes)

def get_model(num_classes=10, dropout_rate=0.2, use_spike=False, T=8, model_type='standard'):
    """Returns the appropriate model based on parameters
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (not used in ResNet34)
        use_spike: Whether to use spiking neurons
        T: Number of time steps for spiking neurons
        model_type: Type of model to use ('standard', 'stateless_resnet')
        
    Returns:
        A neural network model
    """
    if model_type == 'stateless_resnet':
        print(f"Using StatelessResNet34 model with T={T}")
        return get_stateless_resnet18(num_classes=num_classes, T=T)
    else:
        print(f"Using standard ResNet34 model")
        return get_resnet34(num_classes=num_classes)