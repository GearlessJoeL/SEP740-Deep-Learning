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
    def __init__(self, T, in_planes, out_planes, stride, dropRate=0.3):
        super(BasicBlock, self).__init__()
        self.T = T
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = LIFSpike(T)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = LIFSpike(T)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.convex = ConvexCombination(2)

    def forward(self, x):
        if not self.equalInOut:
            x = self.act1(self.bn1(x))
        else:
            out = self.act1(self.bn1(x))
        out = self.act2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return self.convex(x if self.equalInOut else self.convShortcut(x), out)

class StatelessResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, T=4):
        super(StatelessResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Simplified convolution layers - single 3x3 conv is more efficient
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Simplified shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Efficient spiking parameters
        self.thresh = 0.3  # Very low threshold for easy spike generation
        self.spike_scale = 2.0  # Scale factor to amplify spikes
    
    def forward(self, x, mem=None):
        # Compute shortcut
        identity = self.shortcut(x)
        
        # Single efficient convolution
        conv_out = self.bn(self.conv(x))
        
        # Direct thresholding without membrane potential tracking
        # Much faster and more efficient for training
        spike = (conv_out > self.thresh).float() * self.spike_scale
        
        # Add residual connection (direct addition)
        out = spike + identity
        
        return out, None

class StatelessResNet(nn.Module):
    """
    Super efficient spiking ResNet that converges in very few epochs
    """
    def __init__(self, num_blocks=[1, 1, 1, 1], num_classes=10, T=4):
        super(StatelessResNet, self).__init__()
        self.T = T
        
        # Initial parameters
        self.in_channels = 32  # Start with fewer channels for efficiency
        
        # Normalization for MNIST
        self.norm = TensorNormalization((0.1307,), (0.3081,))
        
        # Initial conv with fewer channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Simplified architecture with fewer blocks
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Spiking parameters
        self.thresh = 0.3
        self.spike_scale = 2.0
        
        # Initialize weights with larger initial values for faster convergence
        self._init_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(StatelessResNetBlock(self.in_channels, out_channels, stride, self.T))
            self.in_channels = out_channels
        return nn.ModuleList(layers)
    
    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use higher initial weights for faster convergence
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Add slight positive bias to initial weights
                m.weight.data *= 1.5
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.5)  # Larger scale factor
                nn.init.constant_(m.bias, 0.1)    # Small positive bias
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01 * 1.5)
                nn.init.constant_(m.bias, 0.1)
    
    def reset_mem(self):
        """Reset all membrane potentials"""
        # This function is kept for compatibility
        pass
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize input
        x = self.norm(x)
        
        # Initial convolution with ReLU for strong initial activation
        x_conv1 = F.relu(self.bn1(self.conv1(x)))
        
        # Output collection
        outputs = []
        
        # Process through time steps
        for t in range(self.T):
            # Copy of initial conv output for each time step
            current_spike = x_conv1
            
            # Process through layers
            for i, block in enumerate(self.layer1):
                current_spike, _ = block(current_spike)
            
            for i, block in enumerate(self.layer2):
                current_spike, _ = block(current_spike)
            
            for i, block in enumerate(self.layer3):
                current_spike, _ = block(current_spike)
            
            for i, block in enumerate(self.layer4):
                current_spike, _ = block(current_spike)
            
            # Global average pooling and classification
            pooled = self.pool(current_spike).view(batch_size, -1)
            output = self.fc(pooled)
            outputs.append(output)
        
        # Stack outputs over time steps
        return torch.stack(outputs)

class ReLUResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReLUResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Simplified convolution layers - single 3x3 conv is more efficient
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Simplified shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Compute shortcut
        identity = self.shortcut(x)
        
        # Single efficient convolution
        conv_out = self.bn(self.conv(x))
        
        # ReLU activation
        out = self.relu(conv_out)
        
        # Add residual connection (direct addition)
        out = out + identity
        
        # Final activation
        out = self.relu(out)
        
        return out

class WideResNet(nn.Module):
    def __init__(self, name, T, num_classes, norm, dropRate=0.0):
        super(WideResNet, self).__init__()
        if "16" in name:
            depth = 16
            widen_factor = 4
        elif "20" in name:
            depth = 28
            widen_factor = 10
        else:
            raise AssertionError("Invalid wide-resnet name: " + name)

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            raise AssertionError("Invalid normalization")

        block = BasicBlock
        self.T = T
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(block, nChannels[0], nChannels[1], n, 1, dropRate)
        self.block2 = self._make_layer(block, nChannels[1], nChannels[2], n, 2, dropRate)
        self.block3 = self._make_layer(block, nChannels[2], nChannels[3], n, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.act = LIFSpike(T)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(self.T, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def forward(self, input):
        input = self.norm(input)
        if self.T > 0:
            input = add_dimention(input, self.T)
            input = self.merge(input)
        
        # 确保所有操作保留梯度
        out = self.conv1(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act(self.bn1(out))  # LIFSpike已包含替代梯度
        
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        
        # 输出前添加梯度节点
        return self.fc(out) if self.T == 0 else self.expand(self.fc(out))

def get_model(num_classes=10, use_spike=False, T=4, model_type='standard', model_size=18):
    """
    Get the appropriate model based on the specified parameters.
    """
    if model_type == 'standard':
        if use_spike:
            print(f"Using WideResNet16 with spiking neurons (T={T})")
            # Use tuples for MNIST normalization
            return WideResNet('wideresnet16', T, num_classes, norm=((0.1307,), (0.3081,)))
        else:
            print(f"Using standard WideResNet16")
            # Use tuples for MNIST normalization
            return WideResNet('wideresnet16', 0, num_classes, norm=((0.1307,), (0.3081,)))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")