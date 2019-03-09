import os
import math
from glob import glob
from collections import defaultdict
from typing import List, Optional, Dict, Any, Iterator

import torch
import numpy as np


def generate_best_dataset_index(corpus_dir: str,
                                article_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Modified version of https://github.com/rkcosmos/deepcut/blob/master/deepcut/train.py#L65
    """
    from sklearn.model_selection import train_test_split

    if article_types is None:
        article_types = ["article", "encyclopedia", "news", "novel"]

    dataset_index = defaultdict(list)

    for article_type in article_types:
        files = glob(os.path.join(corpus_dir, article_type, '*.txt'))
        files = [f.replace(corpus_dir+"/", "", 1) for f in files]
        files_train, files_test = train_test_split(files,
                                                   random_state=0,
                                                   test_size=0.1)
        files_train, files_val = train_test_split(files_train,
                                                  random_state=0,
                                                  test_size=0.1)

        dataset_index["train"] += files_train
        dataset_index["validate"] += files_val
        dataset_index["test"] += files_test

    return dataset_index


def reverse_dict(dict: Dict[Any, Any]) -> Dict[Any, Any]:
    return {v: k for k, v in dict.items()}


def pad_sequence(sequences: List[str],
                 max_length: int,
                 overlap: Optional[float] = None,
                 pad_val: int = 0,
                 dtype: type = np.int32,
                 return_len: bool = False) -> Iterator[np.ndarray]:
    if overlap is None:
        overlap = int(max_length * 0.3) + 1

    if max_length <= overlap:
        raise ValueError("max_length <= overlap")

    sequences = np.array(sequences, dtype=dtype)
    stride = max_length - overlap
    seq_len = len(sequences)

    if max_length > seq_len:
        out_seq_count = 1
    else:
        out_seq_count = math.ceil((seq_len - max_length) / stride) + 1

    start_pos = 0

    for i in range(out_seq_count):
        end_pos = start_pos + max_length

        if end_pos > seq_len:
            pad_seq = np.ones(end_pos - seq_len, dtype=dtype) * pad_val
            output = np.concatenate([sequences[start_pos:], pad_seq])
            out_cont_len = len(sequences[start_pos:])
        else:
            output = sequences[start_pos:end_pos]
            out_cont_len = len(sequences[start_pos:end_pos])
        start_pos += stride
        assert len(output) == max_length

        if return_len:
            yield (output, out_cont_len)
        else:
            yield output


def get_params(model: torch.nn.Module, requires_grad: Optional[bool] = None):
    return [p for p in model.parameters()
            if requires_grad is None or p.requires_grad == requires_grad]


def get_summary_writer(log_dir: Optional[str] = None):
    try:
        import os
        import pytz
        import tempfile
        from datetime import datetime

        from tensorboardX import SummaryWriter

        curr_time = datetime \
            .now(tz=pytz.timezone('Asia/Tokyo')) \
            .strftime("%Y%m%d_%H%M")
        if log_dir is None:
            log_dir = tempfile.mkdtemp(prefix=f"{curr_time}_")
        else:
            curr_time = datetime.now(tz=pytz.timezone('Asia/Tokyo')) \
                .strftime("%Y%m%d_%H%M")
            log_dir = os.path.join(log_dir, curr_time)

        print(f"Setting the tensorboard log dir to {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)
        return writer

    except ImportError:
        print("tensorboardX is not available, disable tensorboard")
        return None


def count_parameters(model, requires_grad=None):
    return sum(p.numel() for p in get_params(model, requires_grad))


def model_summary(model):
    all_params_count = count_parameters(model, requires_grad=None)
    trainable_params_count = count_parameters(model, requires_grad=True)

    return (f"{str(model)} \n"
            f"Number of all parameter: {all_params_count:,} \n"
            f"Number of trainable parameter: {trainable_params_count:,}")
