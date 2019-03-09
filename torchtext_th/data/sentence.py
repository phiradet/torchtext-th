from typing import Iterator
from itertools import chain

from torchtext_th.data.token import Token


class Sentence(object):

    def __init__(self, raw_sentence: str, delim: str = "|") -> None:
        self.delim = delim
        self.tokens = []
        for t in raw_sentence.strip().split(delim):
            if len(t) > 0:
                self.tokens.append(Token(t))

    def to_chars(self, is_norm: bool = False) -> Iterator[str]:
        return chain.from_iterable([t.to_chars(is_norm) for t in self.tokens])

    def to_bmes_labels(self)-> Iterator[str]:
        return chain.from_iterable([t.to_bmes_labels() for t in self.tokens])

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return self.delim.join([str(t) for t in self.tokens])
