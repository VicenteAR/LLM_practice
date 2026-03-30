from typing import Dict
import tiktoken


class Tokenizer:
    """
    Basic implementation of Byte Pair Encoding (BPE) tokenizer.

    This class takes a text and encodes it into a list of tokens, or
    decodes a list of unicode values into text.
    """

    def __init__(self):
        self.merge_dict = {}
        self.generate_vocab_dict()

    def get_stats(self, int_list: list, *args, **kwargs) -> Dict:
        """Given a list of integers, return consecutive pairs and their occurrences."""
        stats = {}
        for pairs in zip(int_list, int_list[1:]):
            stats[pairs] = stats.get(pairs, 0) + 1
        # we can arrenge the dictionary
        stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
        return stats

    def merge_tokens(self, int_list: list, pair: set, new_token: int) -> list:
        """Given a list of integers int_list, replace the pair by the new_token value."""
        new_list = []
        i = 0
        while i <= len(int_list) - 1:
            if i != len(int_list) - 1 and (int_list[i], int_list[i + 1]) == pair:
                new_list.append(new_token)
                i += 2
            else:
                new_list.append(int_list[i])
                i += 1
        return new_list

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ):
        """Train the tokenizer with a given text, generating a merge list."""
        # translate the code using UTF-8
        tokens = text.encode("UTF-8")
        tokens = list(map(int, tokens))
        new_tokens = tokens.copy()
        # loop to generate merges according to max byte pair occurrence
        num_merges = vocab_size - 256
        # new merges dictionary
        self.merge_dict = {}
        for ix in range(num_merges):
            # get text statistics to get the most common pair
            pairs = self.get_stats(int_list=new_tokens)
            max_pair = max(pairs, key=pairs.get)
            new_tk = ix + 256
            self.merge_dict[max_pair] = new_tk
            # apply merges
            new_tokens = self.merge_tokens(new_tokens, max_pair, new_tk)
        if verbose:
            print(
                f"Original token length: {len(tokens)}. New token length: {len(new_tokens)}. Comp: {len(tokens) / len(new_tokens):.2f}X"
            )
        # once we have generate the merge_dict object, we can create our vocab
        self.generate_vocab_dict()

    def generate_vocab_dict(self):
        """Generate a dictionary mapping tokens to their corresponding unicode values."""
        self.vocab_dict = {i: bytes([i]) for i in range(256)}
        for (i, j), new_tk in self.merge_dict.items():
            self.vocab_dict[new_tk] = self.vocab_dict[i] + self.vocab_dict[j]

    def check_train(self, verbose: bool = False):
        """
        Review whether the object class has vocab and merge_dict objects.

        The method analyzes that the train method has been used, generating
        self.vocab and self.merge_dict. If not, the function collects the
        information from gpt4 tokenizer and fills the missing attributes.
        """
        if len(self.merge_dict) <= 1:  # we do not have merges
            self.get_tiktoken_merges()  # generate merges from tiktoken library
            # we can generate vocab once we have the merges
            self.generate_vocab_dict()
            message = "Merge dict obtained from tiktoken."
        else:
            message = "Merge dict obtained from train method."
        print(message) if verbose else None

    def decode(self, tokens: list) -> str:
        """Decode a list of tokens into text using the merges dictionary."""
        text = b"".join([self.vocab_dict[ix] for ix in tokens])
        text = text.decode("UTF-8", errors="replace")
        return text

    def encode(self, text: str) -> list:
        """Ëncode a text into a list of tokens using BPE merges."""
        # text is encoded using utf-8
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        while len(tokens) >= 2:
            # get all combinations inside tokens
            stats = self.get_stats(tokens)
            # get the pair inside tokens with the lowest merge index
            pair = min(stats, key=lambda x: self.merge_dict.get(x, float("inf")))
            # we have to replace the pair by the new BPE token
            # if sentence: when there aren't more merges, all key arguments
            # are inf, returning the first occurrence inside stats. To avoid a
            # key error, we check the pair is inside the merge_dict.
            if pair in self.merge_dict.keys():
                ix = self.merge_dict[pair]
                tokens = self.merge_tokens(tokens, pair, ix)
            else:
                break
        return tokens

    def get_tiktoken_ranks(self) -> dict:
        """Get titoken gpt4 mergeable ranks."""
        enc = tiktoken.get_encoding("cl100k_base")
        return enc._mergeable_ranks

    def bpe(self, mergeable_ranks, token, max_rank):
        """
        Helper function used in get_gpt4_merges to reconstruct the merge forest.

        Function obtained from minibpe library.
        """

        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )
        return parts

    def recover_merges(self, mergeable_ranks: dict) -> dict:
        """Function to recover gpt4 merge_dict and vocab_dict."""
        # the `merges` are already the byte sequences in their merged state.
        # so we have to recover the original pairings. We can do this by doing
        # a small BPE training run on all the tokens, in their order.
        # function obtained in minibpe library.
        merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue  # skip raw bytes
            pair = tuple(self.bpe(mergeable_ranks, token, max_rank=rank))
            assert len(pair) == 2
            # recover the integer ranks of the pair
            ix0 = mergeable_ranks[pair[0]]
            ix1 = mergeable_ranks[pair[1]]
            merges[(ix0, ix1)] = rank

        return merges

    def get_tiktoken_merges(self):
        """
        Generate gpt4 merge_dict and vocab_dict.

        This function recovers gpt4 merges and generates merge_dict
        when the latter is missing.
        """
        # get merges from tiktoken library
        mergeable_ranks = self.get_tiktoken_ranks()
        self.merge_dict = self.recover_merges(mergeable_ranks)
