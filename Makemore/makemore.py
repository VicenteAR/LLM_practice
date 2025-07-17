import torch
import torch.nn.functional as F
import random
from typing import Optional


class Biagrams:

    def __init__(self):
        pass

    def _generate_abc(self):
        self.abc = ["*"] + sorted(list(set("".join(self.words))))
        self.stoi = {ch: ix for ix, ch in enumerate(self.abc)}
        self.itos = {ix: ch for ch, ix in self.stoi.items()}

    def open(self, file: str):
        with open(file, "r") as f:
            self.words = f.read().splitlines()
        self._generate_abc()

    def train_model(self, show: bool = True):
        N = torch.zeros(size=(len(self.abc), len(self.abc)))
        for w in self.words:
            w = ["*"] + list(w) + ["*"]
            for ch1, ch2 in zip(w, w[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                N[ix1, ix2] += 1
        # Normalization
        self.P = (N + 1).float()  # we add 1 for model smoothing
        self.P /= N.sum(dim=1, keepdim=True)
        if show:
            return N

    def generate_words(self, num_words: int = 1):
        g = torch.Generator().manual_seed(2147483647)
        word = ["*"]
        for _ in range(num_words):
            ix = 0
            while True:
                p = self.P[ix]
                ix = torch.multinomial(
                    input=p, num_samples=1, replacement=True, generator=g
                ).item()
                ch = self.itos[ix]
                word.append(ch)
                if ch == "*":
                    break
        return "".join(word)

    def return_probabilities(self, word: str):
        word = ["*"] + list(word) + ["*"]
        nll = 0.0
        n = 0
        for ch1, ch2 in zip(word, word[1:]):
            ix1 = self.stoi[ch1]
            ix2 = self.stoi[ch2]
            prob = self.P[ix1, ix2]
            nll += torch.log(prob)
            n += 1
            print(f"{ch1}{ch2}, prob={prob.item():.4f}, log={torch.log(prob):.4f}")
        print(f"neg likelihood={-nll/n}")


class NNBiagrams:

    def __init__(self):
        pass

    def open(self, file: str):
        with open(file, "r") as f:
            self.words = f.read().splitlines()
        self._generate_abc()

    def _generate_abc(self):
        abc = ["*"] + sorted(list(set("".join(self.words))))
        self.stoi = {ch: ix for ix, ch in enumerate(abc)}
        self.itos = {ix: ch for ch, ix in self.stoi.items()}

    def generate_train_model(self, extension: Optional[int] = -999):
        xs = []
        ys = []
        ext = len(self.words) if extension == -999 else extension
        for word in self.words[:ext]:
            w = ["*"] + list(word) + ["*"]
            for ch1, ch2 in zip(w, w[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        return xs, ys

    def initialize_parameters(self, xs):
        self.num = xs.nelement()  # number of train elements (number of biagrams)
        self.dim = len(self.stoi)  # dimension of the problem. Number of characters
        g = torch.Generator().manual_seed(2147483647)  # seed generator
        xenc = F.one_hot(xs, num_classes=self.dim).float()  # one-hot encoding tensor xs
        W = torch.randn(
            size=(self.dim, self.dim), generator=g, requires_grad=True
        )  # weight parameters
        return xenc, W

    def forward_pass(self, xenc, ys):
        logits = xenc @ self.W  # predict logit counts
        counts = logits.exp()  # counts, equivalent to matrix N
        P = counts / counts.sum(
            dim=1, keepdim=True
        )  # probabilities for the next character
        loss = (
            -P[torch.arange(self.num), ys].log().mean() + 0.05 * ((self.W) ** 2).mean()
        )  # loss = negative log likelihood + regularization loss
        # We add the W mean for model smoothing. This results in uniform probabilities (W -> 0, logits -> 0, counts -> 1, P -> uniform)
        return loss

    def backward_pass(self, loss):
        self.W.grad = None
        loss.backward()

    def update_pass(self, lr: float):
        self.W.data += -lr * self.W.grad

    def gradient_descent(self, extension=-999, num_iter: int = 100, lr: float = 0.5):
        # Generate train model
        xs, ys = self.generate_train_model(extension=extension)
        # Initialize parameters
        xenc, self.W = self.initialize_parameters(xs=xs)
        # Gradient descent loop
        for k in range(num_iter):
            loss = self.forward_pass(
                xenc=xenc, ys=ys
            )  # We move 1 step forward through the network. Get probabilities and the loss
            self.backward_pass(
                loss=loss
            )  # We reset the gradients to zero. Move backwards to get the dependency loss(W)
            self.update_pass(lr=lr)  # We update the values of the parameters
            print(f"iteration = {k}. Loss = {loss}")

    def generate_words(self, num_words: int = 1):
        g = torch.Generator().manual_seed(2147483647)
        word = ["*"]
        for _ in range(num_words):
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=self.dim).float()
                counts = (xenc @ self.W).exp()
                p = counts / counts.sum(dim=1, keepdim=True)
                ix = torch.multinomial(
                    input=p, num_samples=1, replacement=True, generator=g
                ).item()
                ch = self.itos[ix]
                word.append(ch)
                if ch == "*":
                    break
        return "".join(word)


class ContextNNBiagrams:

    def __init__(self, context: int):
        self.context = context

    def _generate_abc(self):
        self.abc = ["*"] + sorted(list(set("".join(self.words))))
        self.stoi = {ch: ix for ix, ch in enumerate(self.abc)}
        self.itos = {ix: ch for ch, ix in self.stoi.items()}

    def open(self, file: str):
        with open(file, "r") as f:
            self.words = f.read().splitlines()
        self._generate_abc()
        # randomize the order of the words
        random.seed(42)
        random.shuffle(self.words)

    def generate_train_model(
        self, depth: int = -999
    ) -> tuple[torch.tensor, torch.tensor]:
        X = []
        Y = []
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

    def generate_train_test_sets(self, xs: torch.tensor, ys: torch.tensor):
        n1 = int(0.8 * len(xs))
        n2 = int(0.9 * len(xs))
        self.xtrain, self.xval, self.xtest = torch.tensor_split(xs, (n1, n2), dim=0)
        self.ytrain, self.yval, self.ytest = torch.tensor_split(ys, (n1, n2), dim=0)

    def initialize_parameters(self, emb: int = 10, w1_shape: int = 200):
        context = self.xtrain.shape[
            1
        ]  # Context (number of preceeding elements) of train set
        dim = len(self.stoi)  # dimension of the problem. Number of characters
        g = torch.Generator().manual_seed(2147483647)  # seed generator
        self.C = torch.randn(size=(dim, emb), generator=g).float()  # embedding
        self.W1 = torch.randn(
            size=(context * emb, w1_shape), generator=g, requires_grad=True
        )  # weight first layer
        self.b1 = torch.randn(w1_shape, generator=g)  # bias first layer
        self.W2 = torch.randn(size=(w1_shape, dim), generator=g)  # weight second layer
        self.b2 = torch.randn(dim, generator=g)  # bias second layer
        for p in [self.C, self.W1, self.b1, self.W2, self.b2]:
            p.requires_grad = True

    def forward_pass(self, batch_size, xs, ys, minibatch: bool = True):
        # minibatch
        if minibatch:
            ixs = torch.randint(low=0, high=xs.shape[0], size=(batch_size,))
            xtr = xs[ixs]
            ytr = ys[ixs]
        else:
            xtr = xs
            ytr = ys
            print(f"dim(xtr) = {xtr.shape}. dim(ytr) = {ytr.shape}")
        # embedding
        emb = self.C[xtr]
        X = emb.view(-1, emb.shape[1] * emb.shape[2])
        # first layer
        h = torch.tanh((X @ self.W1) + self.b1)
        # second layer
        logits = (h @ self.W2) + self.b2
        return F.cross_entropy(input=logits, target=ytr)

    def backward_pass(self, loss: torch.tensor):
        for p in [self.C, self.W1, self.b1, self.W2, self.b2]:
            p.grad = None
        loss.backward()

    def update_pass(self, lr):
        for p in [self.C, self.W1, self.b1, self.W2, self.b2]:
            p.data += -lr * p.grad

    def generate_train_val_test_splits(self, depth: int):
        xs, ys = self.generate_train_model(depth=depth)
        self.generate_train_test_sets(xs=xs, ys=ys)

    def gradient_descent_train(self, lr: float, num_iter: int, batch_size: int):
        self.loss_train = []
        for i in range(num_iter):
            # forward pass. Returns loss
            loss = self.forward_pass(
                batch_size=batch_size, xs=self.xtrain, ys=self.ytrain, minibatch=True
            )
            self.loss_train.append(loss)
            # backward pass. Updates grads
            self.backward_pass(loss=loss)
            # update pass. Recalculates params
            lr = lr if i < int(0.75 * num_iter) else lr / 100
            self.update_pass(lr=lr)
            if i % 10000 == 0:
                print(f"Iteration:{i}")

    def generate_words(self, num_words: int = 1):
        g = torch.Generator().manual_seed(2147483647)
        words = ["*"]
        for _ in range(num_words):
            context = [0] * self.context
            while True:
                emb = self.C[torch.tensor(context)]
                X = emb.view(1, -1)
                # first layer
                h = torch.tanh((X @ self.W1) + self.b1)
                # second layer
                logits = (h @ self.W2) + self.b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(
                    input=probs, num_samples=1, replacement=True, generator=g
                ).item()
                ch = self.itos[ix]
                context = context[1:] + [ix]
                words.append(ch)
                if ch == "*":
                    break
        return "".join(words)


# test = ContextNNBiagrams(context=3)
# test.open("names.txt")
# print(test.words)
