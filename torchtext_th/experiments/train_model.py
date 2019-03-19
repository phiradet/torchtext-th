import os
import json
from datetime import datetime
from typing import Dict, List, Iterator, Optional, Any, Tuple

import pandas as pd
import torch
import numpy as np

from torchtext_th.data.corpus import BESTCorpus
from torchtext_th.data.sentence import Sentence
from torchtext_th.data.vocab import Vocab
from torchtext_th.data.label import BMESLabel
from torchtext_th.modules.crf import allowed_transitions
from torchtext_th.modules.bi_lstm_crf import CNNBiLSTMCRF
from torchtext_th.modules.trainer import Trainer
from torchtext_th.tokenizer import save_tokenizer

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def curr_time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def main(corpus_dir: str, corpus_index: Dict[str, List[str]],
         checkpoint_dir: str, output_dir: str, model_conf: Dict[str, Any],
         is_norm: bool = True, min_freq: int = 50,
         log_dir: Optional[str] = None, max_len: int = 200,
         batch_size: int = 64, max_epochs: int = 15, lr: float = 0.001) -> None:

    assert os.path.exists(output_dir)
    assert os.path.exists(corpus_dir)
    assert os.path.exists(checkpoint_dir)

    print(f"Start loading corpus from {corpus_dir}")
    corpus: BESTCorpus = BESTCorpus(corpus_dir=corpus_dir,
                                    corpus_index=corpus_index)

    train_data: List[Sentence] = list(corpus.train)
    val_data: List[Sentence] = list(corpus.validate)

    target_chars: Iterator[str] = Vocab.get_unique_chars(sentences=train_data,
                                                         min_freq=min_freq,
                                                         is_norm=is_norm)
    target_chars = set(target_chars)

    vocab = Vocab().fit(target_chars=list(target_chars))
    label = BMESLabel()
    print(pd.DataFrame(vocab.char2ind.items(), columns=["token", "ind"]))

    constraints: List[Tuple[int]] = allowed_transitions(constraint_type="BMES",
                                                        labels=label.ind2label)

    model_conf["vocab_count"] = len(vocab.char2ind)
    model_conf["label_count"] = len(label.label2ind)
    model_conf["constraints"] = constraints

    model = CNNBiLSTMCRF(**model_conf)

    checkpoint_dir = os.path.join(checkpoint_dir, curr_time())

    trainer = Trainer(model=model,
                      vocab=vocab,
                      label=label,
                      max_len=max_len,
                      log_dir=log_dir,
                      norm_char=is_norm) \
        .fit(train_sentences=train_data,
             batch_size=batch_size,
             epochs=max_epochs,
             lr=lr,
             val_sentences=val_data,
             checkpoint_dir=checkpoint_dir)

    output_artifact = os.path.join(output_dir, f"{curr_time()}_model.pt")
    save_tokenizer(vocab=trainer.vocab,
                   seq_labeler=trainer.model,
                   output_path=output_artifact)
    print(f"Model artifact is stored at {output_artifact}")


if __name__ == "__main__":
    model_conf = dict(embedding_dim=64,
                      lstm_hidden_dim=256,
                      lstm_num_layers=2,
                      p_drop=0.5,
                      p_drop_cnn=0.2,
                      cnn_filter_sizes=[1, 3, 5],
                      cnn_filter_nums=128)
    main(corpus_dir="./corpus/full",
         corpus_index=json.load(open("./corpus/corpus_index.json")),
         checkpoint_dir="./checkpoints",
         output_dir="./artifacts",
         model_conf=model_conf,
         log_dir="./logs",
         batch_size=64)
