from typing import Dict, List

import pytest

from spacy_to_hf import spacy_to_hf
from tests.constants import (
    HF_TAGS_1,
    HF_TAGS_2,
    HF_TOKENS_1,
    HF_TOKENS_2,
    SPACY_DATA_1,
    SPACY_DATA_2,
)


@pytest.mark.parametrize(
    "spacy_data,hf_tokens,hf_tags",
    [
        (SPACY_DATA_1, HF_TOKENS_1, HF_TAGS_1),
        (SPACY_DATA_2, HF_TOKENS_2, HF_TAGS_2),
    ],
)
def test_spacy_to_hf(
    spacy_data: List[Dict], hf_tokens: List[str], hf_tags: List[str]
) -> None:
    hf_data = spacy_to_hf(spacy_data, "bert-base-cased")
    assert hf_data["tokens"][0] == hf_tokens
    assert hf_data["ner_tags"][0] == hf_tags


def test_spacy_to_hf_spans_not_list() -> None:
    spacy_data = [
        {
            "text": "I have a BSc (Bachelors of Computer Sciences) from NYU",
            "spans": dict(start=9, end=12, label="degree"),
        }
    ]
    with pytest.raises(AssertionError) as e:
        spacy_to_hf(spacy_data, "bert-base-cased")  # type: ignore
    assert str(e.value) == "Spans must be a list"


def test_spacy_to_hf_bad_keys() -> None:
    spacy_data = [
        {
            "my_text": "I have a BSc (Bachelors of Computer Sciences) from NYU",
            "spans": [
                dict(start=9, end=12, label="degree"),
                dict(start=14, end=44, label="degree"),
                dict(start=51, end=54, label="university"),
            ],
        }
    ]
    with pytest.raises(AssertionError) as e:
        spacy_to_hf(spacy_data, "bert-base-cased")  # type: ignore
    assert str(e.value) == "All rows must have 2 keys, 'spans', and 'text'"
