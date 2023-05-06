# spacy-to-hf
A simple converter from Spacy Entities to Huggingface BILOU formatted data

I've always struggled to convert my spacy formatted spans into data that can be trained
on using huggingface transformers. But Spacy's Entity format is the most intuitive
format for tagging entities for NER.

This repo is a simple converter that leverages `spacy.gold.biluo_tags_from_offsets`
and the SpaCy `tokenizations` repo that creates a 1-line function to convert spacy
formatted spans to `tokens` and `ner_tags` that can be fed into any
Token Classification Transformer

## Installation
```shell
pip install spacy-to-hf
python -m spacy download en_core_web_sm
````

## Usage
```python
from spacy_to_hf import spacy_to_hf
from datasets import Dataset

span_data = [
    {
        "text": "I have a BSc (Bachelors of Computer Sciences) from NYU",
        "spans": [
            {"start": 9, "end": 12, "label": "degree"},
            {"start": 14, "end": 44, "label": "degree"},
            {"start": 51, "end": 54, "label": "university"}
        ]
    }
]
hf_data = spacy_to_hf(span_data, "bert-base-cased")
print(list(zip(hf_data["tokens"][0], hf_data["ner_tags"][0])))
ds = Dataset.from_dict(hf_data)
```

From here, you can label-index your ner_tags and prepare for fine-tuning

## Project Setup
Project setup is credited to @anthonycorletti and his awesome repo
https://github.com/anthonycorletti/python-project-template
