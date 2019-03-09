from typing import List

from torchtext_th.data.sentence import Sentence


def test_to_char_basic():
    chars: List[str] = list(Sentence("วัน|จันทร์|").to_chars())
    expected = ['ว', 'ั', 'น', 'จ', 'ั', 'น', 'ท', 'ร', '์']
    assert chars == expected


def test_to_char_ner():
    test_case = "|จำ|ได้| |<NE>สิงห์</NE>| |เรา|จำ|ได้|"
    chars: List[str] = list(Sentence(test_case).to_chars())
    expected = ['จ', 'ำ', 'ไ', 'ด', '้', ' ', 'ส', 'ิ', 'ง', 'ห', '์', ' ',
                'เ', 'ร', 'า', 'จ', 'ำ', 'ไ', 'ด', '้']
    assert chars == expected


def test_to_bmew_basic():
    chars: List[str] = list(Sentence("วัน|จันทร์|").to_bmes_labels())
    expected = ['B', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'E']
    assert chars == expected


def test_to_bmew_ner():
    test_case = "|จำ|ได้| |<NE>สิงห์</NE>|"
    chars: List[str] = list(Sentence(test_case).to_bmes_labels())
    expected = ['B', 'E', 'B', 'M', 'E', 'W', 'B', 'M', 'M', 'M', 'E']
    assert chars == expected
