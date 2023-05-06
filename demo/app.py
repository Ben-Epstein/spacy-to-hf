import json
from json import JSONDecodeError

import spacy
import spacy.cli
import streamlit as st
from datasets import Dataset

from spacy_to_hf import spacy_to_hf

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")
    st.experimental_rerun()


demo_option = [
    {
        "text": "Planned to go to the Apple Storefront on Tuesday",
        "spans": [
            {"start": 0, "end": 7, "label": "Action"},
            {"start": 21, "end": 37, "label": "Loc"},
            {"start": 41, "end": 48, "label": "Date"},
        ],
    }
]

tokenizers = [
    "bert-base-uncased",
    "bert-base-cased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "roberta-base",
]
tok = st.selectbox("Pick a tokenizer", tokenizers)
spacy_data = st.text_area("Input your NER Span data here")

if spacy_data or st.button("Or try an example"):
    run_data = None
    if spacy_data:
        try:
            run_data = json.loads(spacy_data)
        except JSONDecodeError as e:
            st.warning(f"Invalid JSON data, try again\n{str(e)}")
    else:
        run_data = demo_option
    if run_data:
        st.write("Spacy input data:")
        st.json(run_data)
        hf_data = spacy_to_hf(run_data, tok)
        df = Dataset.from_dict(hf_data).to_pandas()
        st.write("Output huggingface format:")
        st.dataframe(df)
