# config_models.py

MODELS = {
    "small": {
        "model_id": "text-embedding-3-small",
        "source": "openai"
    },
    "large": {
        "model_id": "text-embedding-3-large",
        "source": "openai"
    },
    "mpnet": {
        "model_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "source": "local"
    },
    "distiluse": {
        "model_id": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "source": "local"
    },
    "minilm": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "source": "local"
    },
    "xlmr": {
        "model_id": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "source": "local"
    },
    "labse": {
        "model_id": "sentence-transformers/LaBSE",
        "source": "local"
    },
    "retrieva": {
        "model_id": "sonoisa/sentence-bert-base-ja-mean-tokens",
        "source": "local"
    },
    "ruri": {
        "model_id": "cl-nagoya/ruri-v3-310m",
        "source": "local"
    },
    "plamo": {
        "model_id": "pfnet/plamo-embedding-1b",
        "source": "local"
    },
    "rosetta": {
        "model_id": "pkshatech/RoSEtta-base-ja",
        "source": "local"
    },
    "sarashina": {
        "model_id": "sbintuitions/sarashina-embedding-v1-1b",
        "source": "local"
    }

}
