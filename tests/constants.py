SPACY_DATA_1 = [
    {
        "text": "I have a BSc (Bachelors of Computer Sciences) from NYU",
        "spans": [
            dict(start=9, end=12, label="degree"),
            dict(start=14, end=44, label="degree"),
            dict(start=51, end=54, label="university"),
        ],
    }
]
HF_TOKENS_1 = [
    "I",
    "have",
    "a",
    "BS",
    "##c",
    "(",
    "Bachelor",
    "##s",
    "of",
    "Computer",
    "Sciences",
    ")",
    "from",
    "NY",
    "##U",
]

HF_TAGS_1 = [
    "O",
    "O",
    "O",
    "B-degree",
    "L-degree",
    "O",
    "B-degree",
    "I-degree",
    "I-degree",
    "I-degree",
    "L-degree",
    "O",
    "O",
    "B-university",
    "L-university",
]


SPACY_DATA_2 = [
    {
        "text": "Planned to go to the Apple Storefront on Tuesday in Plandome",
        "spans": [
            {"start": 0, "end": 7, "label": "Action"},
            {"start": 21, "end": 37, "label": "Loc"},
            {"start": 41, "end": 48, "label": "Date"},
            {"start": 52, "end": 60, "label": "Loc"},
        ],
    }
]

HF_TOKENS_2 = [
    "Plan",
    "##ned",
    "to",
    "go",
    "to",
    "the",
    "Apple",
    "Store",
    "##front",
    "on",
    "Tuesday",
    "in",
    "Plan",
    "##dom",
    "##e",
]
HF_TAGS_2 = [
    "B-Action",
    "L-Action",
    "O",
    "O",
    "O",
    "O",
    "B-Loc",
    "I-Loc",
    "L-Loc",
    "O",
    "U-Date",
    "O",
    "B-Loc",
    "I-Loc",
    "L-Loc",
]
