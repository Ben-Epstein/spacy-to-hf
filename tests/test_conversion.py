from typing import Dict, List

import pytest
from datasets import Dataset

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


@pytest.mark.parametrize(
    "spacy_data,hf_tokens,hf_tags",
    [
        (SPACY_DATA_1, HF_TOKENS_1, HF_TAGS_1),
        (SPACY_DATA_2, HF_TOKENS_2, HF_TAGS_2),
    ],
)
def test_spacy_to_hf_as_dataset(
    spacy_data: List[Dict], hf_tokens: List[str], hf_tags: List[str]
) -> None:
    hf_data = spacy_to_hf(spacy_data, "bert-base-cased", as_hf_dataset=True)
    hf_non_o_tags = [i for i in hf_tags if i != "O"]
    sorted_tags = ["O"] + sorted(set(hf_non_o_tags))
    assert isinstance(hf_data, Dataset)
    assert hf_data.features["ner_tags"].feature.names == sorted_tags
    assert hf_data["tokens"][0] == hf_tokens
    assert hf_data["ner_tags"][0] == [sorted_tags.index(tag) for tag in hf_tags]


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
