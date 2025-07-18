from typing import Optional
import torch


class Linear:
    """Create a Linear layer"""

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        generator: Optional[int],
        bias: bool = True,
    ):
        """Initialization of the class."""
        self.g = torch.Generator().manual_seed(generator) if generator else None
        self.weight = (
            torch.randn(size=(fan_in, fan_out), generator=self.g) / fan_in**0.5
        )
        self.bias = torch.randn(fan_out, generator=self.g) if bias else None

    def __call__(self, x: torch.Tensor):
        """Perform linear W @ x operation."""
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        """Return list of parameters (weights and bias if any)."""
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class BatchNorm1D:
    """Create a Batch Normalization layer. This class renormalizes the output of a Linear Layer class"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """Initialize the class."""
        self.eps = eps
        self.momentum = momentum
        self.train = True
        # backpropagation parameters
        self.bngain = torch.ones(num_features)
        self.bnbias = torch.zeros(num_features)
        # buffers (trained with running momentum update)
        with torch.no_grad():
            self.bnmean_running = torch.zeros(num_features)
            self.bnvar_running = torch.ones(num_features)

    def __call__(self, hpreact=torch.Tensor):
        """Perform Batch Normalization operation over a Linear Layer income."""
        # Mean and variance
        if self.train == True:
            mean = hpreact.mean(dim=0, keepdim=True)
            var = hpreact.var(dim=0, keepdim=True)
            # During training, update running mean and variance
            with torch.no_grad():
                self.bnmean_running = (
                    1.0 - self.momentum
                ) * self.bnmean_running + self.momentum * mean
                self.bnvar_running = (
                    1.0 - self.momentum
                ) * self.bnvar_running + self.momentum * var
        else:
            # No training. Use running mean and variance for inference
            mean = self.bnmean_running
            var = self.bnvar_running
        # Compute output
        out = (hpreact - mean) / torch.sqrt(var + self.eps) * self.bngain + self.bnbias
        return out

    def parameters(self):
        return [self.bngain, self.bnbias]


class Tanh:
    """Create a non-linear (tanh) layer."""

    def __init__(self):
        """Initialize the class."""
        pass

    def __call__(self, x: torch.Tensor):
        """Apply a non-linear (tanh) over a Linear class output."""
        return torch.tanh(x)

    def parameters(self):
        return []
