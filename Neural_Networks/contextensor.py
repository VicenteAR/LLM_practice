import random
import torch


class ContextTorchTensor:
    """
    Transform a name list into pytorch tensors.

    This class takes a txt file containing names (or any string list) and
    transforms it into a set X, Y pytorch tensors. Each character is
    displayed as a integer. For example, 'a' -> 1, 'b' -> 2, etc.
    The tensors are 'context'-dimensional, meaning that each row contains
    n elements, n given by the selected context.
    """

    def __init__(self, context) -> None:
        self.context = context
        self.vocab_size = None

    def _generate_abc(self):
        self.abc = ["*"] + sorted(list(set("".join(self.words))))
        self.stoi = {ch: ix for ix, ch in enumerate(self.abc)}
        self.itos = {ix: ch for ch, ix in self.stoi.items()}
        # Define the size of the dictionary. This is given by the number of characters.
        self.vocab_size = len(self.stoi)

    def open(self, file: str):
        with open(file, "r") as f:
            self.words = f.read().splitlines()
        self._generate_abc()
        # randomize the order of the words
        random.seed(42)
        random.shuffle(self.words)

    def get_tensors(self, depth: int = -999) -> tuple[torch.Tensor, torch.Tensor]:
        X = []
        Y = []
        # extension of the problem. How many words the module takes from the words attribute.
        ext = len(self.words) if depth == -999 else depth
        for word in self.words[:ext]:
            context = [0] * self.context
            for ch in list(word) + ["*"]:
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y


class TensorSplit:
    """Split a set X,Y torch tensors into train/validation/test sets."""

    def __init__(self, train_size: float, test_size: float):
        """Initialize the class."""
        assert isinstance(train_size, float)
        assert isinstance(test_size, float)
        self.train_size = train_size
        self.test_size = train_size + test_size
        assert self.train_size + self.test_size <= 1

    def split(self, xs: torch.Tensor, ys: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Split the incoming xs, ys torch tensors into train/val/test sets."""
        # Define dimensions train/test/validation splits
        n1 = int(self.train_size * len(xs))
        n2 = int(self.test_size * len(xs))
        # Generate splits
        xtrain, xval, xtest = torch.tensor_split(xs, (n1, n2), dim=0)
        ytrain, yval, ytest = torch.tensor_split(ys, (n1, n2), dim=0)
        return xtrain, xval, xtest, ytrain, yval, ytest
