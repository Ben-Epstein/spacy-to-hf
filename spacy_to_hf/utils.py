from itertools import chain
from typing import Dict, List

from datasets import ClassLabel, Dataset, Sequence


def next_token_is_same(tokens: List[List[int]], cur_idx: int, tok_num: int) -> bool:
    if cur_idx >= len(tokens) - 1:
        return False
    next_idx = cur_idx + 1
    return tok_num == tokens[next_idx][0]


def prev_token_is_same(tokens: List[List[int]], cur_idx: int, tok_num: int) -> bool:
    if cur_idx == 0:
        return False
    prev_idx = cur_idx - 1
    return tok_num == tokens[prev_idx][-1]


def _is_unit_tag(tag: str) -> bool:
    return tag != "O" and tag.split("-")[0] == "U"


def _is_begin_tag(tag: str) -> bool:
    return tag != "O" and tag.split("-")[0] == "B"


def _is_last_tag(tag: str) -> bool:
    return tag != "O" and tag.split("-")[0] == "L"


def _get_label(tag: str) -> str:
    return tag.split("-")[1]


def _handle_unit_tag(
    tag: str, tokens: List[List[int]], cur_idx: int, tok_num: int
) -> str:
    """Process a Unit tag

    If a Unit tagged token is broken into multiple sub-tokens, we want the first
    to become a B, all middles to be I, and the final one to be an L
    """
    prev_tok_match = prev_token_is_same(tokens, cur_idx, tok_num)
    next_tok_match = next_token_is_same(tokens, cur_idx, tok_num)
    clean_tag = tag
    if prev_tok_match and next_tok_match:
        clean_tag = f"I-{_get_label(tag)}"
    elif prev_tok_match:
        clean_tag = f"L-{_get_label(tag)}"
    elif next_tok_match:
        clean_tag = f"B-{_get_label(tag)}"
    return clean_tag


def _handle_begin_tag(
    tag: str, tokens: List[List[int]], cur_idx: int, tok_num: int
) -> str:
    """Process a Begin tag

    For Begin tagged tokens that are broken into sub-tokens, we know that there will be
    all I tokens (or a single L token) after the first instance. Therefore, we can
    check if we are the first instance of this B tag, otherwise we become an I tag
    """
    clean_tag = tag
    if prev_token_is_same(tokens, cur_idx, tok_num):
        clean_tag = f"I-{_get_label(tag)}"
    return clean_tag


def _handle_last_tag(
    tag: str, tokens: List[List[int]], cur_idx: int, tok_num: int
) -> str:
    """Process a Last tag

    For Last tagged tokens that are broken into sub-tokens, we know that there will be
    all I tokens (or a single B token) before the first instance. Therefore, we can
    simply check if we are the last instance of this L tag, otherwise we become an I tag
    """
    clean_tag = tag
    if next_token_is_same(tokens, cur_idx, tok_num):
        clean_tag = f"I-{_get_label(tag)}"
    return clean_tag


def map_spacy_to_hf_tags(
    hf_to_spacy: List[List[int]], spacy_tags: List[str]
) -> List[str]:
    """Maps the spacy_tags to the required huggingface tags

    Leverages the hf_to_spacy map, showing how each huggingface token maps
    its corresponding spacy token.

    Ex:
        text = "I have a BSc (Bachelors of Computer Sciences) from NYU"
        spacy_tokens = ['I', 'have', 'a', 'BSc', '(', 'Bachelors', 'of', 'Computer',
                        'Sciences', ')', 'from', 'NYU']
        entities = Entities: [(9, 12, ORG), (14, 44, ORG)]
        spacy_bilou = ['O', 'O', 'O', 'U-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'L-ORG',
                                                                        'O', 'O', 'O']

        bert_tokens = ['i', 'have', 'a', 'bsc', '(', 'bachelor', '##s', 'of', 'computer'
                                                    , 'sciences', ')', 'from', 'nyu']
        bert_to_spacy = [[0], [1], [2], [3], [4], [5], [5], [6], [7], [8], [9], [10],
                                                                                [11]]

        map_spacy_to_hf_tags(bert_to_spacy, spacy_tags)
            -> ['O', 'O', 'O', 'U-ORG', 'O', 'B-ORG', 'L-ORG', 'I-ORG', 'I-ORG', 'L-ORG'
                                                                    , 'O', 'O', 'O']
    """
    hf_tags = []
    for cur_idx, hf_spacy_tokens in enumerate(hf_to_spacy):
        mapped_hf_tags = [spacy_tags[i] for i in hf_spacy_tokens]
        clean_hf_tags = []
        for tag, tok_num in zip(mapped_hf_tags, hf_spacy_tokens):
            # When converting from spacy to the hf tokenizer, it's possible spacy tokens
            # are further broken down into sub-tokens. If this happens, a B- tag,
            # U- tag, or L- tag may be duplicated. If this is the case, we need to
            # correct this issue by converting it to the proper tag-type depending on
            # the next and previous tokens seen
            clean_tag = tag
            if _is_unit_tag(tag):
                clean_tag = _handle_unit_tag(tag, hf_to_spacy, cur_idx, tok_num)
            elif _is_begin_tag(tag):
                clean_tag = _handle_begin_tag(tag, hf_to_spacy, cur_idx, tok_num)
            elif _is_last_tag(tag):
                clean_tag = _handle_last_tag(tag, hf_to_spacy, cur_idx, tok_num)

            clean_hf_tags.append(clean_tag)

        hf_tags.extend(clean_hf_tags)
    return hf_tags


def dict_to_dataset(hf_data: Dict[str, List[str]]) -> Dataset:
    """Converts a dictionary of huggingface data into a well-formed Dataset

    ex input:
        {
            "tokens": [["sentence", "1"], ["sentence", "Apple"]],
            "ner_tags": [["U-word", "O"], ["U-word", "U-ORG"]]
        }

    This will create a huggingface dataset from the input, and also map the `ner_tags`
    into a ClassLabel object which is required for training.
    """
    labels = sorted(set(chain.from_iterable(hf_data["ner_tags"])))
    # O is typically the first tag. Move it there
    if "O" in labels:
        labels.remove("O")
        labels.insert(0, "O")
    ds = Dataset.from_dict(hf_data)
    # https://github.com/python/mypy/issues/6239
    class_label = Sequence(feature=ClassLabel(num_classes=len(labels), names=labels))
    # First need to string index the ner_tags
    label_to_idx = dict(zip(labels, range(len(labels))))
    ds = ds.map(
        lambda row: {"ner_tags": [label_to_idx[tag] for tag in row["ner_tags"]]}
    )
    # Then we can create the ClassLabel
    ds = ds.cast_column("ner_tags", class_label)
    return ds
