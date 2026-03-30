from tokenizer import Tokenizer
from typing import Dict, Optional
import regex as re


class RegexTokenizer(Tokenizer):
    """
    Implementation of a regex-based tokenizer.

    This class extends the basic BPE tokenizer
    by using regular expressions to split the text into tokens.
    """

    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self):
        super().__init__()
        self.pattern = re.compile(self.GPT4_SPLIT_PATTERN)

    def get_stats(self, int_list: list, stats=None, *args, **kwargs) -> Dict:
        """Given a list of integers, return consecutive pairs and their occurrences."""
        stats = {} if stats == None else stats
        for pairs in zip(int_list, int_list[1:]):
            stats[pairs] = stats.get(pairs, 0) + 1
        return stats

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ):
        """
        Train the tokenizer with a given text, generating a merge list.

        The text is split following the gpt4 pattern. Each chunk is encoded
        using utf8 and analyzed to get most common pairs.
        """
        split_text = re.findall(self.pattern, text)
        # translate the code using UTF-8
        tokens = [list(map(int, ch.encode("utf-8"))) for ch in split_text]
        new_tokens = (
            tokens.copy()
        )  # instead of having a list of integers, now we have a list of chunks
        # loop to generate merges according to max byte pair occurrence
        num_merges = vocab_size - 256
        # new merges dictionary
        self.merge_dict = {}
        # iteration over the desired number of merges
        for ix in range(num_merges):
            stats = {}
            # as we have a list of chunks, we must iterate over
            # the different chunks and store the results
            for tk in new_tokens:
                # get text statistics to get the most common pair
                stats = self.get_stats(int_list=tk, stats=stats)
            # we rearrange the stats
            stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
            max_pair = max(stats, key=stats.get)
            new_tk = ix + 256
            self.merge_dict[max_pair] = new_tk
            # apply merges
            new_tokens = [self.merge_tokens(tk, max_pair, new_tk) for tk in new_tokens]
        if verbose:
            tokens_len = 0
            new_tokens_len = 0
            for ix in range(len(tokens)):
                tokens_len += len(tokens[ix])
                new_tokens_len += len(new_tokens[ix])
            print(
                f"Original token length: {tokens_len}. New token length: {new_tokens_len}. Comp: {tokens_len / new_tokens_len:.2f}X"
            )
        # once we have generate the merge_dict object, we can create our vocab
        self.generate_vocab_dict()

    def _encode(self, text: str) -> list:
        """Ëncode a text into a list of tokens using BPE merges."""
        # text is encoded using utf-8
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        while len(tokens) >= 2:
            # get all combinations inside tokens
            stats = self.get_stats(tokens, stats=None)
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

    def encode(self, text: str) -> list:
        """
        Encode the given text using the merge_dict attribute.

        The text is split into chunks of text by categories defined in regex pattern.
        """
        split_text = re.findall(self.pattern, text)  # split text into str chunks
        encoded_list = []  # capsule where we store the encoded chunks
        for chunk in split_text:
            print(chunk)
            tk = self._encode(text=chunk)
            encoded_list.extend(tk)
        return encoded_list
