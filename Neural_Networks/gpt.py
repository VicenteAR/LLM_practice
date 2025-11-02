import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainingText:
    """Process text input for training the Neural Network."""

    def __init__(self, input_file: str):
        """Initialize the class"""
        with open(file=input_file, mode="r", encoding="utf-8") as f:
            self.text = f.read()
        # Call method to get the unique character list
        self._generate_abc()
        # Call method to generate token dictionaries
        self._generate_maps()
        # Compute the vocabulary length
        self.vocab_size = len(self.stoi)
        # Create enconder/decoder
        self._generate_encoders()

    def _generate_abc(self):
        """Get unique characters from the input text"""
        self.abc = list(sorted(set(self.text)))

    def _generate_maps(self):
        """
        Create maps to translate integers into characters and viceversa.

        This method is used to generate the tokenizer map. The tokenization
        is done at character level.
        """
        self.stoi = {ch: i for i, ch in enumerate(self.abc)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def _generate_encoders(self):
        """
        Define enconder/decoder to translate characters into tokens (and viceversa).

        This method defines two attributes (encoder and decoder) that return a list
        of tokens given a series of characters or a list of characters given a list
        of tokens.
        """
        self.encoder = lambda s: [
            self.stoi[ch] for ch in s
        ]  # encoder: returns a list of tokens given a string
        self.decoder = lambda t: "".join(
            [self.itos[tk] for tk in t]
        )  # decoder: returns a string given a list of integers


class SetSplit:
    """Split data torch tensor into train/validation/test sets."""

    def __init__(self, train_size: float, test_size: float):
        """Initialize the class."""
        assert isinstance(train_size, float)
        assert isinstance(test_size, float)
        self.train_size = train_size
        self.test_size = train_size + test_size
        assert train_size + test_size <= 1

    def split(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the incoming data torch tensor into train/val/test sets."""
        # Define dimensions train/test/validation splits
        n1 = int(self.train_size * len(data))
        n2 = int(self.test_size * len(data))
        # Generate splits
        train, val, test = torch.tensor_split(data, (n1, n2), dim=0)
        return train, val, test


class GetBatch:
    """Generate batch tensors (input x, target y) from the given data."""

    def __init__(self, batch_size, context, device) -> None:
        self.batch_size = batch_size
        self.context = context
        self.device = device

    def __call__(self, data):
        ix = torch.randint(
            low=0, high=len(data) - self.context, size=(self.batch_size,)
        )
        x = torch.stack([data[t : t + self.context] for t in ix], dim=0)
        y = torch.stack([data[t + 1 : t + self.context + 1] for t in ix], dim=0)
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class BiagramLanguageModel(nn.Module):
    """Creates a LLM based on the simple Biagram NN architechure."""

    def __init__(self, vocab_size):
        super().__init__()  # call the superclass to inherit its methods
        # Creates embedding layer.
        # Each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, input, target=None):
        """Iterates one step through the NN, creating the grad attribute."""
        # Output of the NN
        logits = self.embedding(input)  # This creates a B, T, C tensor
        if target == None:
            loss = None
        else:
            # To compute the loss function using cross entropy, pytorch needs a
            # B , C , T tensor
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # we generate a B*T, C tensor, where C (embeeding dim) = vocab size
            target = target.view(B * T)
            # Compute loss function
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, input, num_iterations):
        """Generate new tokens based on the given input."""
        for _ in range(num_iterations):
            logits, loss = self(input)
            # We only need the last logit through T dimension
            logits = logits[:, -1, :]  # Creates a B, C tensor (C = vocab size)
            probs = F.softmax(logits, dim=1)  # B, C
            next_char = torch.multinomial(input=probs, num_samples=1)  # B, 1
            input = torch.cat((input, next_char), dim=1)
        return input


class EstimateLoss:
    """Averages the model loss each eval iterations."""

    def __init__(self, train_data, val_data):
        self.train = train_data
        self.val = val_data

    @torch.no_grad()
    def calculate_loss(self, model, get_batch, eval_iter):
        out = {}
        model.eval()  # Set the model in eval mode
        for mode in ["train", "val"]:
            data = self.train if mode == "train" else self.val  # Define data set
            losses = torch.zeros(eval_iter)  # Generate tensor to store the losses
            for k in range(eval_iter):
                xb, yb = get_batch(
                    data=data
                )  # Get a batch tensor from the data set (train/val)
                logits, loss = model(
                    xb, yb
                )  # Call the model and compute logits and losses
                losses[k] = loss.item()  # extract the data from tensor
            out[mode] = losses.mean()  # Compute the mean for the eval_iter elements
        model.train()  # Set again the model in train mode
        return out
