"""Utility functions for gradient-based NAS.

This module provides helper functions for DARTS and ENAS implementations including:
- GPU management utilities
- Parameter counting
- Learning rate scheduling
- Drop path regularization
- Architecture visualization

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def check_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device(use_cuda: bool = True) -> 'torch.device':
    """
    Get PyTorch device (CUDA or CPU).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device object
        
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model: 'nn.Module') -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
        
    Example:
        >>> from torchvision import models
        >>> model = models.resnet18()
        >>> params = count_parameters(model)
        >>> print(f"Parameters: {params:,}")
    """
    if not TORCH_AVAILABLE:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: 'nn.Module') -> Dict[str, int]:
    """
    Count parameters for each layer in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    if not TORCH_AVAILABLE:
        return {}
    
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = param.numel()
    
    return param_dict


def get_model_size_mb(model: 'nn.Module') -> float:
    """
    Estimate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def drop_path(
    x: 'torch.Tensor',
    drop_prob: float,
    training: bool = True
) -> 'torch.Tensor':
    """
    Drop path (Stochastic Depth) regularization.
    
    Randomly drops entire paths during training with probability drop_prob.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path
        training: Whether in training mode
        
    Returns:
        Tensor with paths potentially dropped
        
    Reference:
        Huang et al. "Deep Networks with Stochastic Depth." ECCV 2016.
        
    Example:
        >>> x = torch.randn(32, 128, 16, 16)
        >>> out = drop_path(x, drop_prob=0.2, training=True)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    if not training or drop_prob == 0.:
        return x
    
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize
    
    output = x.div(keep_prob) * random_tensor
    return output


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    
    Example:
        >>> losses = AverageMeter()
        >>> for batch in data_loader:
        ...     loss = compute_loss(batch)
        ...     losses.update(loss.item(), batch_size)
        >>> print(f"Average loss: {losses.avg:.4f}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def accuracy(output: 'torch.Tensor', target: 'torch.Tensor', topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model predictions (logits), shape (N, num_classes)
        target: Ground truth labels, shape (N,)
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
        
    Example:
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> top1, top5 = accuracy(logits, targets, topk=(1, 5))
    """
    if not TORCH_AVAILABLE:
        return [0.0] * len(topk)
    
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    
    Gradually decreases learning rate following a cosine curve.
    
    Args:
        optimizer: PyTorch optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(self, optimizer: 'torch.optim.Optimizer', T_max: int, eta_min: float = 0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
    
    def step(self):
        """Update learning rate."""
        import math
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                 (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            param_group['lr'] = lr
        
        self.last_epoch += 1
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        state: State dictionary containing model, optimizer, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        
    Example:
        >>> state = {
        ...     'epoch': epoch,
        ...     'model_state': model.state_dict(),
        ...     'optimizer_state': optimizer.state_dict(),
        ...     'best_acc': best_acc
        ... }
        >>> save_checkpoint(state, 'checkpoint.pth.tar', is_best=True)
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot save checkpoint")
        return
    
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    
    if is_best:
        import shutil
        best_path = filepath.replace('.pth.tar', '_best.pth.tar')
        shutil.copyfile(filepath, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(filepath: str, model: 'nn.Module', optimizer: Optional['torch.optim.Optimizer'] = None) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        
    Returns:
        Checkpoint dictionary
        
    Example:
        >>> checkpoint = load_checkpoint('checkpoint.pth.tar', model, optimizer)
        >>> start_epoch = checkpoint['epoch']
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        
    Example:
        >>> set_seed(42)
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def print_model_summary(model: 'nn.Module', input_size: Tuple[int, ...]):
    """
    Print model summary including layer shapes and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Example:
        >>> model = MyModel()
        >>> print_model_summary(model, (3, 32, 32))
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available")
        return
    
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        logger.warning("torchsummary not installed. Install with: pip install torchsummary")
        # Fallback: simple parameter count
        total_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")


def freeze_model(model: 'nn.Module'):
    """
    Freeze all model parameters (set requires_grad=False).
    
    Args:
        model: PyTorch model
    """
    if not TORCH_AVAILABLE:
        return
    
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Model parameters frozen")


def unfreeze_model(model: 'nn.Module'):
    """
    Unfreeze all model parameters (set requires_grad=True).
    
    Args:
        model: PyTorch model
    """
    if not TORCH_AVAILABLE:
        return
    
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Model parameters unfrozen")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / (1024 ** 2),
        'cached': torch.cuda.memory_reserved() / (1024 ** 2),
        'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def print_memory_usage():
    """Print current GPU memory usage."""
    mem = get_memory_usage()
    if mem:
        print(f"GPU Memory: Allocated={mem['allocated']:.2f}MB, "
              f"Cached={mem['cached']:.2f}MB, "
              f"Max={mem['max_allocated']:.2f}MB")
    else:
        print("GPU not available or PyTorch not installed")


def get_lr(optimizer: 'torch.optim.Optimizer') -> List[float]:
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        List of current learning rates
    """
    if not TORCH_AVAILABLE:
        return []
    return [group['lr'] for group in optimizer.param_groups]


def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradient norm of parameters.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum norm
        norm_type: Type of norm (2 for L2)
        
    Returns:
        Total norm before clipping
    """
    if not TORCH_AVAILABLE:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        mode: 'min' to minimize metric, 'max' to maximize
        delta: Minimum change to qualify as improvement
        
    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(epochs):
        ...     val_acc = validate(...)
        ...     if early_stopping(val_acc):
        ...         print("Early stopping triggered")
        ...         break
    """
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better than best."""
        if self.mode == 'min':
            return score < self.best_score - self.delta
        else:
            return score > self.best_score + self.delta


# Version check
def check_pytorch_version(min_version: str = "1.7.0") -> bool:
    """
    Check if PyTorch version meets minimum requirement.
    
    Args:
        min_version: Minimum required version
        
    Returns:
        True if version is sufficient
    """
    if not TORCH_AVAILABLE:
        return False
    
    from packaging import version
    current_version = torch.__version__.split('+')[0]  # Remove +cu111 suffix
    
    return version.parse(current_version) >= version.parse(min_version)


if __name__ == "__main__":
    print("Gradient-Based NAS Utilities")
    print("=" * 50)
    
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {check_cuda_available()}")
        
        if check_cuda_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print_memory_usage()
        
        # Test utilities
        print("\nTesting utilities...")
        meter = AverageMeter()
        for i in range(10):
            meter.update(i, 1)
        print(f"Average meter test: avg={meter.avg:.2f}")
        
        print("\nAll utilities loaded successfully!")
    else:
        print("PyTorch not available. Install with: pip install torch")
