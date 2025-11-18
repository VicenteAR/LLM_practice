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


class SingleHeadBiagramLanguageModel(nn.Module):
    """Creates a LLM based on the Biagram NN with single-head architechure."""

    def __init__(self, vocab_size, context_size, emb_size, head_size):
        super().__init__()  # call the superclass to inherit its methods
        self.context_size = context_size
        # Creates embedding layers.
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_emb_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size
        )
        self.position_emb_table = nn.Embedding(
            num_embeddings=context_size, embedding_dim=emb_size
        )
        # Creates single head of self-attention. Returns a tensor where past tokens are averaged.
        self.head = Head(
            context_size=context_size, emb_size=emb_size, head_size=head_size
        )
        # Creates linear layer to produce logits
        self.lm_nn = nn.Linear(in_features=head_size, out_features=vocab_size)

    def forward(self, input, target=None):
        """Iterates one step through the NN, creating the grad attribute."""
        # Dimensions
        B, T = input.shape
        # Input embedding
        tk_emb = self.token_emb_table(input)  # (B,T,C)
        pos_emb = self.position_emb_table(torch.arange(T))  # (T,C)
        x = tk_emb + pos_emb  # (B,T,C)
        # Apply one-head of Self-attention
        x = self.head(x)  # (B,T,H)
        # Output of the NN
        logits = self.lm_nn(x)  # This creates a (B,T,vocab size) tensor
        # Compute the loss
        if target == None:
            loss = None
        else:
            # To compute the loss function using cross entropy, pytorch needs a
            # B , C , T tensor
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # we generate a B*T, C tensor, where C = vocab size
            target = target.view(B * T)
            # Compute loss function
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, input, num_iterations):
        """Generate new tokens based on the given input."""
        for _ in range(num_iterations):
            # As we use a position embedding, we must crop the input
            input_enc = input[:, -self.context_size :]
            logits, loss = self(input_enc)
            # We only need the last logit through T dimension
            logits = logits[
                :, -1, :
            ]  # takes the last token. Creates a (B,C) tensor (C = vocab size)
            probs = F.softmax(logits, dim=1)  # B, C
            next_char = torch.multinomial(input=probs, num_samples=1)  # B, 1
            input = torch.cat((input, next_char), dim=1)
        return input


class MultiHeadBiagramLanguageModel(nn.Module):
    """Creates a LLM based on multi-head self-attention architechure."""

    def __init__(self, vocab_size, context_size, emb_size, head_size, head_dim):
        super().__init__()  # call the superclass to inherit its methods
        self.context_size = context_size
        # Creates embedding layers.
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_emb_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size
        )
        self.position_emb_table = nn.Embedding(
            num_embeddings=context_size, embedding_dim=emb_size
        )
        # Creates multi-head self-attention block.
        self.head = MultiHead(context_size, emb_size, head_size // head_dim, head_dim)
        # Creates linear layer to produce logits
        self.lm_nn = nn.Linear(in_features=head_size, out_features=vocab_size)

    def forward(self, input, target=None):
        """Iterates one step through the NN, creating the grad attribute."""
        # Dimensions
        B, T = input.shape
        # Input embedding
        tk_emb = self.token_emb_table(input)  # (B,T,C)
        pos_emb = self.position_emb_table(torch.arange(T))  # (T,C)
        x = tk_emb + pos_emb  # (B,T,C)
        # Apply one-head of Self-attention
        x = self.head(x)  # (B,T,H)
        # Output of the NN
        logits = self.lm_nn(x)  # This creates a (B,T,vocab size) tensor
        # Compute the loss
        if target == None:
            loss = None
        else:
            # To compute the loss function using cross entropy, pytorch needs a
            # B , C , T tensor
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # we generate a B*T, C tensor, where C = vocab size
            target = target.view(B * T)
            # Compute loss function
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, input, num_iterations):
        """Generate new tokens based on the given input."""
        for _ in range(num_iterations):
            # As we use a position embedding, we must crop the input
            input_enc = input[:, -self.context_size :]
            logits, loss = self(input_enc)
            # We only need the last logit through T dimension
            logits = logits[
                :, -1, :
            ]  # takes the last token. Creates a (B,C) tensor (C = vocab size)
            probs = F.softmax(logits, dim=1)  # B, C
            next_char = torch.multinomial(input=probs, num_samples=1)  # B, 1
            input = torch.cat((input, next_char), dim=1)
        return input


class Head(nn.Module):
    """Creates a single head of self-attention."""

    def __init__(self, context_size: int, emb_size: int, head_size: int):
        super().__init__()  # call super methods
        # attributes
        self.head_size = head_size
        self.query = nn.Linear(in_features=emb_size, out_features=head_size, bias=False)
        self.key = nn.Linear(in_features=emb_size, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=emb_size, out_features=head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_size, context_size)))
        )

    def forward(self, input):
        """Iterates one step through the single head of self-attention."""
        # Input dimensions
        B, T, C = input.shape
        # Output of the single head
        q = self.query(input)  # (B,T,H)
        k = self.key(input)  # (B,T,H)
        v = self.value(input)  # (B,T,H)
        w = q @ k.transpose(1, 2) * self.head_size**-0.5  # (B,T,H) @ (B,H,T) = (B,T,T)
        w = w.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B,T,T) without speaking with the future
        # Note: we crop self.tril to self.tril[:T, :T] because the input dimension T can vary from 1 to T.
        # The original w dimensions are (b,t,t) where b, t are the actual input dimensions, not the ones given
        # by the parameters.
        w = F.softmax(w, dim=2)  # (B,T,T)
        out = w @ v  # (B,T,T) @ (B,T,H) = (B,T,H)
        return out


class MultiHead(nn.Module):
    """Create a module based on multiple, concatenated Head modules."""

    def __init__(self, context_size, emb_size, head_size, head_dim):
        super().__init__()
        # we create head_dim communication channels in paralel, each one of size head_size
        self.head_list = nn.ModuleList(
            [Head(context_size, emb_size, head_size) for _ in range(head_dim)]
        )

    def forward(self, input):
        # Concatenate the output of the multiple head modules over the channel dimension.
        return torch.cat([h(input) for h in self.head_list], dim=2)


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
