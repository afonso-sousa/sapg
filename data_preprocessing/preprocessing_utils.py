import re

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()


def _normalize_pretokenized(sentence):
    # This expression will match sequences like . . ., ...., . . . . ., and convert them all to "...".
    sentence = re.sub(r"\.{1}\s*\.\s*\.+", "...", sentence)
    normalized_sentence = detokenizer.detokenize(sentence.split())
    return normalized_sentence


def _remove_outer_quotes(sentence):
    if (sentence.startswith("'") and sentence.endswith("'")) or (
        sentence.startswith('"') and sentence.endswith('"')
    ):
        return sentence[1:-1]
    else:
        return sentence


def _remove_whitespaces(sentence):
    # remove leading and trailing spaces
    sentence = sentence.strip()
    # remove double spaces
    sentence = re.sub(" +", " ", sentence)
    return sentence


def _normalize_special_chars(sentence):
    sentence = (
        sentence.replace("``", '"')
        .replace("` `", '"')
        .replace("‘‘", '"')
        .replace("‘ ‘", '"')
        .replace("’’", '"')
        .replace("’ ’", '"')
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("''", '"')
        .replace("' '", '"')
    )
    return sentence


def clean_sentence(sentence):
    sentence = _remove_whitespaces(sentence)
    sentence = _normalize_special_chars(sentence)
    sentence = _remove_outer_quotes(sentence)
    sentence = _normalize_pretokenized(sentence)
    return sentence
