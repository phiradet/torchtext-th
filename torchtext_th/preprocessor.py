from torchtext_th.character_dict import *


def normalize_character(char: str) -> str:
    if char in ENG_LOWER:
        return "<EN_LO>"
    elif char in ENG_UPPER:
        return "<EN_UP>"
    elif char in NUM_ARABIC:
        return "<NUM_AR>"
    elif char in NUM_TH:
        return "<NUM_TH>"
    elif char in PUNC_TH:
        return "<PUN_TH>"
    elif char in ["ๅ", "า"]:
        return "า"
    else:
        return char
