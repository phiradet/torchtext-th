import re
from enum import IntEnum
from typing import List, Iterator

from torchtext_th.preprocessor import normalize_character


class TokenType(IntEnum):
    TYPICAL = 0
    POEM = 1
    NER = 2
    ABBRE = 3


class Token(object):

    tags = ["<NE>", "</NE>", "<AB>", "</AB>", "<POEM>", "</POEM>"]
    escaped_tag = [re.escape(t) for t in tags]
    pattern = re.compile("|".join(escaped_tag))

    def __init__(self, value: str) -> None:
        self.content: str = __class__.remove_tag(value)
        self.t_type: TokenType = __class__.get_type(value)

    @staticmethod
    def remove_tag(value: str) -> str:
        return __class__.pattern.sub("", value)

    @staticmethod
    def get_type(value: str) -> TokenType:
        """

        Args:
            value: the value of token which could contains a tag
                ex. "<NE>กรมควบคุมโรค</NE>" or simply "ประวัติ"

        Returns:
            token type derived from the tag

        """
        if value.startswith("<NE>"):
            return TokenType.NER
        elif value.startswith("<AB>"):
            return TokenType.ABBRE
        elif value.startswith("POEM"):
            return TokenType.POEM
        else:
            return TokenType.TYPICAL

    def to_chars(self, is_norm: bool = False) -> List[str]:
        """
        Convert this token to a sequence of characters
        """
        if is_norm:
            return [normalize_character(c) for c in self.content]
        return list(self.content)

    def to_bmes_labels(self) -> Iterator[str]:
        """
        Convert this token to a sequence of BMES labels:
            B(egine), M(iddle), E(nd), S(ingle)
        """
        token_len = len(self)
        if token_len == 1:
            yield "S"
        else:
            for i in range(token_len):
                if i == 0:
                    label = "B"
                elif i == token_len - 1:
                    label = "E"
                else:
                    label = "M"
                yield label

    def __len__(self):
        return len(self.content)

    def __str__(self):
        return self.content
