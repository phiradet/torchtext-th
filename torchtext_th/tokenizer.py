from typing import List

import torch
from torch import nn

from torchtext_th.data.vocab import Vocab
from torchtext_th.data.sentence import Sentence
from torchtext_th.data.label import Label, BMESLabel
from torchtext_th.modules.bi_lstm_crf import BiLSTMCRF


class Tokenizer(object):

    def __init__(self, vocab: Vocab, seq_labeler: nn.Module,
                 label: Label = None, is_norm: bool = True):
        self.vocab = vocab
        if label is None:
            self.label = BMESLabel()
        else:
            self.label = label
        self.seq_labeler = seq_labeler
        self.is_norm = is_norm

    @staticmethod
    def interpret_tags(sentence: Sentence, tags: List[str]):
        assert tags[-1] == "E" or tags[-1] == "S"
        orig_chars = list(sentence.to_chars(is_norm=False))
        assert len(orig_chars) == len(tags)
        output = []
        agg_char = []
        for char, tag in zip(orig_chars, tags):
            if (tag == "B" or tag == "S") and len(agg_char) > 0:
                output.append("".join(agg_char))
                agg_char = []
            agg_char.append(char)
        if len(agg_char) > 0:
            output.append("".join(agg_char))
        return output

    def tokenize(self, sentence: str, greedy: bool = False) -> List[str]:
        sentence = Sentence(sentence)
        chars = list(sentence.to_chars(is_norm=self.is_norm))
        char_count = len(chars)
        x: torch.Tensor = self.vocab.transform([chars], max_len=char_count)

        input_dict = dict(
            x=x,
            x_length=torch.tensor([char_count])
        )

        with torch.no_grad():
            output = self.seq_labeler(input_dict)

        if greedy:
            emission_logit = output["emission_logit"][0]
            _, max_inds = emission_logit.max(dim=1)
            best_path_ind = list(max_inds.data.numpy())

            # labels = ["<P>", "B", "M", "E", "S"]
            # for c, score in zip(chars, emission_logit):
            #     print("อ"+c)
            #     score = list(score.data.numpy())
            #     print(list(zip(labels, score)))
        else:
            best_path_ind = output["predicted_tags"][0]
        best_path = [self.label.ind2label[t] for t in best_path_ind]
        return __class__.interpret_tags(sentence, best_path)

    def save(self, output_path: str):
        save_tokenizer(
            vocab=self.vocab,
            seq_labeler=self.seq_labeler,
            output_path=output_path
        )


def save_tokenizer(vocab: Vocab, seq_labeler: BiLSTMCRF, output_path: str) -> None:
    model_conf = seq_labeler.kwargs
    tok_dict = dict(
        vocab=vocab.__dict__,
        model_weight=seq_labeler.state_dict(),
        model_conf=model_conf
    )
    torch.save(tok_dict, output_path)


def get_tokenizer(artifact_path: str) -> Tokenizer:
    tok_info = torch.load(artifact_path)

    if "vocab" not in tok_info or \
            "model_weight" not in tok_info or \
            "model_conf" not in tok_info:
        raise ValueError(f"The given model artifact file is wrong! "
                         f"{artifact_path}")
    vocab = Vocab()
    vocab.__dict__ = tok_info["vocab"]
    label = BMESLabel()

    model_conf = tok_info["model_conf"]
    seq_labeler = BiLSTMCRF(**model_conf)
    seq_labeler.load_state_dict(tok_info['model_weight'])
    seq_labeler = seq_labeler.eval()
    return Tokenizer(
        vocab=vocab,
        seq_labeler=seq_labeler,
        label=label,
        is_norm=True
    )

