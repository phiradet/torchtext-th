from itertools import chain
from collections import defaultdict
from typing import List, Dict, Iterator

import torch

from torchtext_th.data.sentence import Sentence
from torchtext_th.utils import reverse_dict, pad_sequence


class Vocab(object):

    def __init__(self):
        self.char2ind: Dict[str, int] = {}
        self.ind2char: Dict[int, str] = {}

    def fit(self, target_chars: List[str]) -> 'Vocab':
        self.char2ind = __class__._get_character_map(target_chars)
        self.ind2char = reverse_dict(self.char2ind)
        return self

    def _encode(self, chars: List[List[str]]) -> Iterator[List[int]]:
        _UNK_ind = self.char2ind["_UNK_"]
        for instance in chars:
            ind_vec = [self.char2ind.get(i, _UNK_ind) for i in instance]
            yield ind_vec

    def decode(self, indexes: List[List[int]],
               ignore_pad: bool = False) -> List[List[str]]:
        output = []
        for instance in indexes:
            label_vec: List[str] = []
            for i in instance:
                str_label = self.ind2char[i]
                if not (ignore_pad and str_label == "_PAD_"):
                    label_vec.append(str_label)
            output.append(label_vec)
        return output

    def transform(self, chars: List[List[str]], max_len: int) -> torch.Tensor:
        X = []
        pad_val = self.char2ind["_PAD_"]
        for ind_vec in self._encode(chars):
            for partitioned_label_vec in pad_sequence(ind_vec,
                                                      max_len,
                                                      pad_val=pad_val):
                X.append(partitioned_label_vec)
        return torch.tensor(X, dtype=torch.long)

    @staticmethod
    def _get_character_map(target_chars: List[str]) -> Dict[str, int]:
        sorted_target_chars = sorted(target_chars)
        sorted_target_chars = ["_PAD_", "_UNK_"] + sorted_target_chars
        return dict(zip(sorted_target_chars, range(len(sorted_target_chars))))

    @staticmethod
    def get_unique_chars(sentences: List[Sentence],
                         min_freq: int,
                         is_norm: bool = True) -> Iterator[str]:
        char_seqs: List[List[str]] = [s.to_chars(is_norm) for s in sentences]
        flatten_vocabs: Iterator[str] = chain.from_iterable(char_seqs)
        vocab_count = defaultdict(int)
        for v in flatten_vocabs:
            vocab_count[v] += 1

        for vocab, freq in vocab_count.items():
            if freq >= min_freq:
                yield vocab

    def __len__(self):
        return len(self.char2ind)
