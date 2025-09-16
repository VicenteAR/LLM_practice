from typing import Optional
import torch


class Embedding:
    """Create an Embedding layer."""

    def __init__(
        self,
        vocab_size: int,
        embeding_dimension: int,
        manual_seed: Optional[int] = None,
    ) -> None:
        """Initialization of the class."""
        g = torch.Generator().manual_seed(g) if manual_seed else manual_seed
        self.weight = torch.randn(size=(vocab_size, embeding_dimension), generator=g)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform embedding operation.

        Given an input tensor of shape (batch size, context size),
        return an output tensor of shape (batch size, context size,
        embedding dimension).

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, context size).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, context size,
                embedding dimension).
        """
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        """Return list of parameters (embedding tensor)."""
        return [self.weight]


class Flatten:
    """Create a Flatten layer. Return a 2D tensor after an Embedding layer."""

    def __init__(self):
        pass

    def __call__(self, x):
        """Reduce the dimensionality of the given tensor."""
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    """Flat an Embedding layer to generate a (B, T, C) tensor."""

    def __init__(self, number_elements: int):
        self.n = number_elements

    def __call__(self, x):
        """
        Flat the incoming (B, C , E) torch tensor.

        Given a torch tensor with (batch size, context, embedding) dimensions,
        the method returns a (batch size, context // n, embedding * n) tensor,
        where n is the number of elements in a pair.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, context size,
                embedding).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, context size // n,
                embedding dimension * n).
        """
        x = x.view(x.shape[0], x.shape[1] // self.n, x.shape[2] * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


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
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

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
        self.bngain = torch.ones(num_features, requires_grad=True)
        self.bnbias = torch.zeros(num_features, requires_grad=True)
        # buffers (trained with running momentum update)
        with torch.no_grad():
            self.bnmean_running = torch.zeros(num_features)
            self.bnvar_running = torch.ones(num_features)

    def __call__(self, hpreact=torch.Tensor):
        """Perform Batch Normalization operation over a Linear Layer income."""
        # Mean and variance
        if self.train == True:
            if hpreact.ndim == 2:
                mean = hpreact.mean(dim=0, keepdim=True)
                var = hpreact.var(dim=0, keepdim=True)
            elif hpreact.ndim == 3:
                mean = hpreact.mean(dim=(0, 1), keepdim=True)
                var = hpreact.var(dim=(0, 1), keepdim=True)
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
        self.out = (hpreact - mean) / torch.sqrt(
            var + self.eps
        ) * self.bngain + self.bnbias
        return self.out

    def parameters(self):
        return [self.bngain, self.bnbias]


class Tanh:
    """Create a non-linear (tanh) layer."""

    def __init__(self):
        """Initialize the class."""
        pass

    def __call__(self, x: torch.Tensor):
        """Apply a non-linear (tanh) over a Linear class output."""
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Layers:
    """Collect a series of NN layers"""

    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequentially the elements inside self.layers to the input x."""
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]
