import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorNormalization(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self, X):
        return normalizex(X, self.mean, self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    # Handle scalar mean and std by converting to proper shape
    if mean.ndim == 0:
        mean = mean.reshape(1)
    if std.ndim == 0:
        std = std.reshape(1)
        
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input

class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=0.5, surrogate='sigmoid', alpha=4.0):
        super().__init__()
        self.T = T
        self.thresh = thresh
        self.tau = tau  # 膜电位衰减系数
        self.alpha = alpha  # 替代梯度陡峭度
        
        # 选择替代梯度函数
        if surrogate == 'sigmoid':
            self.surrogate = lambda x: torch.sigmoid(alpha * x)
        else:  # 默认使用矩形窗
            self.surrogate = lambda x: (x.abs() < 0.5).float()
        
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)

    def forward(self, x):
        if self.T == 0:  # 非脉冲模式
            return F.relu(x)
            
        # 展开时间维度 [B*T, ...] -> [T, B, ...]
        x = self.expand(x)
        batch_size = x.shape[1]
        device = x.device
        
        # 初始化膜电位和脉冲序列
        mem = torch.zeros(batch_size, *x.shape[2:], device=device)
        spikes = []
        
        for t in range(self.T):
            # 更新膜电位 (带梯度)
            mem = mem * self.tau + x[t]
            
            # 生成脉冲（前向：硬阈值，反向：替代梯度）
            spike = (mem > self.thresh).float()
            if self.training or mem.requires_grad:
                spike = spike + (self.surrogate(mem - self.thresh) - self.surrogate(mem - self.thresh).detach())
            
            # 软重置（保留梯度）
            mem = mem - spike * self.thresh
            
            spikes.append(spike)
        
        # 合并时间维度 [T, B, ...] -> [B*T, ...]
        return self.merge(torch.stack(spikes, dim=0))


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out
