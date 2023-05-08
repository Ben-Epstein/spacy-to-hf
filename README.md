# spacy-to-hf
A simple converter from SpaCy Entities (Spans) to Huggingface BILOU formatted data (tokens and ner_tags)

I've always struggled to convert my spacy formatted spans into data that can be trained
on using huggingface transformers. But Spacy's Entity format is the most intuitive
format for tagging entities for NER.

This repo is a simple converter that leverages `spacy.gold.biluo_tags_from_offsets`
and the SpaCy [`tokenizations`](https://github.com/explosion/tokenizations) repo that
creates a 1-line function to convert spacy
formatted spans to `tokens` and `ner_tags` that can be fed into any
Token Classification Transformer

## Try before you buy
You can demo the functionality on [streamlit](https://ben-epstein-spacy-to-hf-demoapp-3u5okj.streamlit.app/) or [spaces](https://huggingface.co/spaces/ben-epstein/ner-spans-to-tokens-tags)
<!-- <iframe src="https://ben-epstein-spacy-to-hf-demoapp-3u5okj.streamlit.app"></iframe> -->

[![Try the app](https://user-images.githubusercontent.com/22605641/236641444-01860522-6caf-4948-82e3-c878fa4616ec.png)](https://huggingface.co/spaces/ben-epstein/ner-spans-to-tokens-tags)



## What is "Spacy" or "HuggingFace" format?
Spacy format simply means having a text input and character level span assignments. <br>
For example:
```python
text = "Hello, my name is Ben"
spans = [{"start": 18, "end": 21, "label": "person"}, ...]
```

This is the common structure of output data from labeling tools like LabelStudio or LabelBox, because it's easy and human interpretable.

Huggingface format refers to the BIO/BILOU/BIOES tagging format commonly used for fine-tuning transformers. The input text is tokenized, and each token
is given a tag to denote whether or not it's a label (and it's location, Beginning, Inside etc). Here's an example: https://huggingface.co/datasets/wikiann
<img width="591" alt="image" src="https://user-images.githubusercontent.com/22605641/236639209-031c6645-e67d-43dc-8d38-be39868d2cd3.png">

For more information about this tagging system, see [wikipedia](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))


This format is tricky, though, because it is entirely dependant on the tokenizer used. Tokens are not simply space separated words. Each tokenizer has a specific vocabulary of tokens that break down works into unique sub-words. So moving from character level spans to token level tags is a very
manual process. That's a core reason I built this tool.

## Installation
```shell
pip install spacy-to-hf
python -m spacy download en_core_web_sm
````

## Usage
```python
from spacy_to_hf import spacy_to_hf

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
```

Or, if you want to immediately start fine-tuning or upload this to huggingface, you can
run
```python
ds = spacy_to_hf(span_data, "bert-base-cased", as_hf_dataset=True)

print(ds.features["ner_tags"].feature.names)
```
This will return your data as a HuggingFace `Dataset` and will automatically
string-index your `ner_tags` into a `ClassLabel` object

## Project Setup
Project setup is credited to [@anthonycorletti](https://github.com/anthonycorletti) and his awesome [project template repo](https://github.com/anthonycorletti/python-project-template)
