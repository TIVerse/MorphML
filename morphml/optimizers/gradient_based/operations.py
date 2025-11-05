"""Operation primitives for gradient-based NAS (DARTS, ENAS).

This module provides efficient operation implementations used in differentiable
architecture search. All operations are PyTorch modules optimized for GPU execution.

Operations:
- Separable Convolutions (SepConv)
- Dilated Convolutions (DilConv)
- Pooling operations (Max, Avg)
- Skip connections (Identity)
- Zero operations (None)

Reference:
    Liu, H., et al. "DARTS: Differentiable Architecture Search." ICLR 2019.
    Pham, H., et al. "Efficient Neural Architecture Search via Parameter Sharing." ICML 2018.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""


try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Dummy classes for type checking
    class nn:
        class Module:
            pass


from morphml.logging_config import get_logger

logger = get_logger(__name__)


def check_torch_available() -> None:
    """Check if PyTorch is available and raise error if not."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gradient-based NAS. "
            "Install with: pip install torch or poetry add torch"
        )


class SepConv(nn.Module):
    """
    Separable Convolution operation.

    Separable convolutions reduce computational cost by factorizing a standard
    convolution into depthwise and pointwise convolutions.

    Standard conv: O(C_in * C_out * k^2)
    Separable conv: O(C_in * k^2 + C_in * C_out)

    Structure:
        ReLU → Depthwise Conv → Pointwise Conv → BatchNorm

    Args:
        C_in: Input channels
        C_out: Output channels
        kernel_size: Kernel size for depthwise conv
        stride: Stride for convolution
        padding: Padding for convolution
        affine: Whether to use learnable affine params in BN

    Example:
        >>> sep_conv = SepConv(16, 32, kernel_size=3, stride=1, padding=1)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = sep_conv(x)  # Shape: (2, 32, 32, 32)
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ):
        super().__init__()

        check_torch_available()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            # Depthwise convolution (groups=C_in)
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            # Pointwise convolution (1x1)
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass."""
        return self.op(x)


class DilConv(nn.Module):
    """
    Dilated (Atrous) Convolution operation.

    Dilated convolutions increase receptive field without increasing
    parameter count by inserting spaces (dilation) between kernel elements.

    Receptive field = k + (k-1) * (dilation-1)

    Structure:
        ReLU → Dilated Depthwise Conv → Pointwise Conv → BatchNorm

    Args:
        C_in: Input channels
        C_out: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding (should be = dilation * (kernel_size - 1) / 2)
        dilation: Dilation rate
        affine: Whether to use learnable affine params in BN

    Example:
        >>> dil_conv = DilConv(16, 32, kernel_size=3, stride=1,
        ...                     padding=2, dilation=2)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = dil_conv(x)  # Shape: (2, 32, 32, 32)
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ):
        super().__init__()

        check_torch_available()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            # Dilated depthwise convolution
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            # Pointwise convolution
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass."""
        return self.op(x)


class Identity(nn.Module):
    """
    Identity operation (skip connection).

    Simply passes input to output unchanged. Used for residual connections.

    Example:
        >>> identity = Identity()
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = identity(x)
        >>> assert torch.equal(x, out)
    """

    def __init__(self):
        super().__init__()
        check_torch_available()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass (identity)."""
        return x


class Zero(nn.Module):
    """
    Zero operation (no connection).

    Returns a zero tensor of the same shape as input. Used to represent
    the absence of a connection in the architecture.

    Args:
        stride: Stride for output shape calculation

    Example:
        >>> zero = Zero(stride=1)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = zero(x)
        >>> assert torch.all(out == 0)
    """

    def __init__(self, stride: int = 1):
        super().__init__()
        check_torch_available()
        self.stride = stride

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass (zeros)."""
        if self.stride == 1:
            return x.mul(0.0)
        else:
            # Stride > 1: reduce spatial dimensions
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """
    Factorized reduction operation.

    Reduces spatial dimensions by factor of 2 while doubling channels.
    More efficient than strided convolution for this specific task.

    Method:
        1. Two parallel 1x1 convolutions with different offsets
        2. Concatenate outputs
        3. BatchNorm

    Args:
        C_in: Input channels
        C_out: Output channels (typically 2 * C_in)
        affine: Whether to use learnable affine params in BN

    Example:
        >>> reduce = FactorizedReduce(16, 32)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = reduce(x)  # Shape: (2, 32, 16, 16)
    """

    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        super().__init__()

        check_torch_available()

        assert C_out % 2 == 0, "C_out must be divisible by 2"

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass."""
        x = self.relu(x)
        # Two parallel convolutions with offset
        out1 = self.conv_1(x)
        out2 = self.conv_2(x[:, :, 1:, 1:])  # Offset by 1 pixel
        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Module):
    """
    Standard convolution block: ReLU → Conv → BatchNorm.

    Args:
        C_in: Input channels
        C_out: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        affine: Whether to use learnable affine params in BN

    Example:
        >>> conv_block = ReLUConvBN(16, 32, 3, 1, 1)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = conv_block(x)  # Shape: (2, 32, 32, 32)
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ):
        super().__init__()

        check_torch_available()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass."""
        return self.op(x)


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) regularization.

    Randomly drops entire paths (operations) during training to prevent
    over-reliance on specific paths and improve generalization.

    Reference:
        Huang et al. "Deep Networks with Stochastic Depth." ECCV 2016.

    Args:
        drop_prob: Probability of dropping a path

    Example:
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = drop_path(x)  # Randomly zeroed during training
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        check_torch_available()
        self.drop_prob = drop_prob

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass with stochastic depth."""
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Create binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (N, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        # Scale and apply mask
        output = x.div(keep_prob) * random_tensor
        return output


# Operation factory function
def create_operation(
    op_name: str, C_in: int, C_out: int, stride: int = 1, affine: bool = True
) -> nn.Module:
    """
    Factory function to create operations by name.

    Args:
        op_name: Operation name
        C_in: Input channels
        C_out: Output channels
        stride: Stride
        affine: Use affine params in BN

    Returns:
        Operation module

    Raises:
        ValueError: If operation name is unknown

    Example:
        >>> op = create_operation('sep_conv_3x3', C_in=16, C_out=32)
        >>> x = torch.randn(2, 16, 32, 32)
        >>> out = op(x)
    """
    check_torch_available()

    if op_name == "none":
        return Zero(stride=stride)

    elif op_name == "skip_connect":
        if stride == 1:
            return Identity()
        else:
            return FactorizedReduce(C_in, C_out, affine=affine)

    elif op_name == "max_pool_3x3":
        return nn.MaxPool2d(3, stride=stride, padding=1)

    elif op_name == "avg_pool_3x3":
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    elif op_name == "sep_conv_3x3":
        return SepConv(C_in, C_out, 3, stride, 1, affine=affine)

    elif op_name == "sep_conv_5x5":
        return SepConv(C_in, C_out, 5, stride, 2, affine=affine)

    elif op_name == "sep_conv_7x7":
        return SepConv(C_in, C_out, 7, stride, 3, affine=affine)

    elif op_name == "dil_conv_3x3":
        return DilConv(C_in, C_out, 3, stride, 2, dilation=2, affine=affine)

    elif op_name == "dil_conv_5x5":
        return DilConv(C_in, C_out, 5, stride, 4, dilation=2, affine=affine)

    elif op_name == "conv_3x3":
        return ReLUConvBN(C_in, C_out, 3, stride, 1, affine=affine)

    elif op_name == "conv_1x1":
        return ReLUConvBN(C_in, C_out, 1, stride, 0, affine=affine)

    else:
        raise ValueError(f"Unknown operation: {op_name}")


# Standard operation set for DARTS/ENAS
OPERATIONS = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


def get_operation_names() -> list:
    """Get list of available operation names."""
    return OPERATIONS.copy()


def count_operation_parameters(op: nn.Module) -> int:
    """
    Count number of trainable parameters in an operation.

    Args:
        op: PyTorch module

    Returns:
        Number of trainable parameters
    """
    if not TORCH_AVAILABLE:
        return 0
    return sum(p.numel() for p in op.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("Testing operations...")

        # Test each operation
        x = torch.randn(2, 16, 32, 32)

        for op_name in OPERATIONS:
            op = create_operation(op_name, C_in=16, C_out=16)
            out = op(x)
            params = count_operation_parameters(op)
            print(f"{op_name:20s} | Output shape: {tuple(out.shape)} | Params: {params:,}")

        print("\nAll operations tested successfully!")
    else:
        print("PyTorch not available. Install with: pip install torch")
