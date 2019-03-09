import os
from typing import Dict, List, Iterator, Union

from tqdm.auto import tqdm

from torchtext_th.data.sentence import Sentence


class BESTCorpus(object):

    def __init__(self, corpus_dir: str,
                 corpus_index: Dict[str, List[str]]) -> None:
        self.corpus_dir = corpus_dir
        self.corpus_index = corpus_index
        self.train = self.read_train_data()
        self.validate = self.read_validate_data()
        self.test = self.read_test_data()

    def read_train_data(self) -> Iterator[Sentence]:
        return self \
            .read_labeled_files(self.corpus_index.get("train", []))

    def read_test_data(self) -> Iterator[Sentence]:
        return self \
            .read_labeled_files(self.corpus_index.get("test", []))

    def read_validate_data(self) -> Iterator[Sentence]:
        return self \
            .read_labeled_files(self.corpus_index.get("validate", []))

    def read_labeled_files(self,
                           paths: Union[str, List[str]],
                           delim: str = "|") -> Iterator[Sentence]:
        if isinstance(paths, str):
            paths = [paths]

        for p in tqdm(paths):
            with open(os.path.join(self.corpus_dir, p)) as f:
                for line in f:
                    line = line.strip()
                    sentence = Sentence(raw_sentence=line, delim=delim)
                    if len(sentence) > 0:
                        yield sentence
