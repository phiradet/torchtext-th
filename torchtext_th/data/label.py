from typing import Dict, List
from abc import ABC, abstractmethod

import torch

from torchtext_th.utils import reverse_dict, pad_sequence


class Label(ABC):

    @abstractmethod
    def _encode(self, labels: List[List[str]]) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indexes: List[List[int]],
               ignore_pad: bool = False) -> List[List[str]]:
        raise NotImplementedError

    def transform(self, labels: List[List[str]], max_len) -> torch.Tensor:
        y = []
        pad_val = self.label2ind["_PAD_"]
        for label_vec in self._encode(labels):
            for partitioned_label_vec in pad_sequence(label_vec,
                                                      max_len,
                                                      pad_val=pad_val):
                y.append(partitioned_label_vec)
        return torch.tensor(y, dtype=torch.long)

    @staticmethod
    def _get_label_map(label_schema: str) -> Dict[str, int]:
        if label_schema != "BMES":
            raise NotImplementedError("Only support BMES")
        else:
            return dict(_PAD_=0, B=1, M=2, E=3, S=4)


class BMESLabel(Label):

    label_schema = "BMES"

    def __init__(self):
        self.label2ind = __class__._get_label_map(__class__.label_schema)
        self.ind2label = reverse_dict(self.label2ind)

    def decode(self, indexes: List[List[int]],
               ignore_pad: bool = False) -> List[List[str]]:
        output: List[List[str]] = []

        for instance in indexes:
            label_vec: List[str] = []
            for i in instance:
                str_label = self.ind2label[i]
                if not (ignore_pad and str_label == "_PAD_"):
                    label_vec.append(str_label)
            output.append(label_vec)
        return output

    def _encode(self, labels: List[List[str]]) -> List[List[int]]:
        output: List[List[int]] = []
        for instance in labels:
            ind_vec: List[int] = [self.label2ind[i] for i in instance]
            output.append(ind_vec)
        return output

    def __len__(self):
        return len(self.label2ind)
