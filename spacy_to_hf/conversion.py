from typing import Collection, Dict, List, Sequence, Union

import spacy
from datasets import Dataset
from spacy.gold import biluo_tags_from_offsets
from tokenizations import get_alignments
from transformers import AutoTokenizer

from spacy_to_hf.utils import dict_to_dataset, map_spacy_to_hf_tags


def spacy_to_hf(
    spacy_data: List[Dict[str, Sequence[Collection[str]]]],
    tokenizer: str,
    as_hf_dataset: bool = False,
) -> Union[Dataset, Dict[str, List[List[str]]]]:
    """Maps spacy formatted spans to HF tokens in BILOU format

    Input should be a list of dictionaries of 'text' and 'spans' keys
    Ex:
        spacy_data = [
            {
                "text": "I have a BSc (Bachelors of Computer Sciences) from NYU",
                "spans": [
                    {"start": 9, "end": 12, "label": "degree"},
                    {"start": 14, "end": 44, "label": "degree"},
                    {"start": 51, "end": 54, "label": "university"}
                ]
            },
            ...
        ]

    Returns a Dictionary with 2 keys: 'tokens', and 'ner_tags' with the tokens provided
        by the given `tokenizer` and the ner_tags created by aligning the spacy tokens
        to the given tokenizer's tokens

        For the above input, we would see:
            hf_data = spacy_to_hf(spacy_data, "bert-base-cased")
            list(zip(hf_data["tokens"][0], hf_data["ner_tags"][0]))
                ->
                    [('I', 'O'),
                     ('have', 'O'),
                     ('a', 'O'),
                     ('BS', 'B-degree'),
                     ('##c', 'L-degree'),
                     ('(', 'O'),
                     ('Bachelor', 'B-degree'),
                     ('##s', 'I-degree'),
                     ('of', 'I-degree'),
                     ('Computer', 'I-degree'),
                     ('Sciences', 'L-degree'),
                     (')', 'O'),
                     ('from', 'O'),
                     ('NY', 'B-university'),
                     ('##U', 'L-university')]


    :param spacy_data: The spacy formatted span data. Must be a list containing
        "text" key and "spans" key. "spans" must be a list of dictionaries with
        "start", "end", and "label"
    :param tokenizer: The tokenizer/model you will be training with in huggingface.
        A good option could be "bert-base-uncased"
    :param as_hf_dataset: If this should return a formatted Huggingface Dataset. If
        True, the dataset will have `tokens` and `ner_tags` as columns, and `ner_tags`
        will be a ClassLabel
    """
    assert all(
        sorted(row.keys()) == ["spans", "text"] for row in spacy_data
    ), "All rows must have 2 keys, 'spans', and 'text'"
    tok = AutoTokenizer.from_pretrained(tokenizer)
    nlp = spacy.load("en_core_web_sm")
    hf_data: Dict[str, List] = {"tokens": [], "ner_tags": []}
    for row in spacy_data:
        spans = row["spans"]
        assert isinstance(spans, list), "Spans must be a list"
        assert all(
            isinstance(span, dict) and sorted(span.keys()) == ["end", "label", "start"]
            for span in spans
        ), "All spans must have keys 'start', 'end', and 'label'"
        text = row["text"]
        doc = nlp(text)
        spacy_tokens = [token.text for token in doc]
        entities = [(span["start"], span["end"], span["label"]) for span in spans]
        spacy_tags = biluo_tags_from_offsets(doc, entities)
        hf_tokens = tok.tokenize(text)
        _, hf_to_spacy = get_alignments(spacy_tokens, hf_tokens)
        hf_tags = map_spacy_to_hf_tags(hf_to_spacy, spacy_tags)
        hf_data["tokens"].append(hf_tokens)
        hf_data["ner_tags"].append(hf_tags)
    if as_hf_dataset:
        return dict_to_dataset(hf_data)
    return hf_data
