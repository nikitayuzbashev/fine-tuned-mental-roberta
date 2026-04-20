"""
classifier.py
=============
Loads the fine-tuned MentalRoBERTa model from HuggingFace Hub and exposes
a single predict() function used by the rest of the RAG pipeline.

Usage in the pipeline
---------------------
    from src.classifier import predict

    result = predict("I feel completely empty and can't get out of bed.")
    print(result["label"])    # "Depression"
    print(result["scores"])   # {"Normal": 0.02, "Anxiety": 0.05, ...}

The model is downloaded from HuggingFace on first use and cached locally
at ~/.cache/huggingface/ — subsequent calls load from cache instantly.

HuggingFace authentication (one-time, only needed if the repo is private)
--------------------------------------------------------------------------
    pip install huggingface_hub
    huggingface-cli login     # paste a READ token from hf.co/settings/tokens
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# ============================================================
# CONFIGURATION
# ============================================================

# Update this to your HuggingFace repo name after uploading
HF_MODEL_REPO = "nyuzbashev/mental-roberta-finetuned"

LABEL_MAP = {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
ID2LABEL  = {v: k for k, v in LABEL_MAP.items()}
MAX_LENGTH = 512

# ============================================================
# DEVICE DETECTION
# ============================================================
if _TORCH_AVAILABLE and torch.cuda.is_available():
    _device = torch.device("cuda")
elif _TORCH_AVAILABLE and torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")

# ============================================================
# LAZY MODEL LOADING
# Model and tokeniser are loaded once on first call to predict()
# and reused for all subsequent calls — no repeated downloads.
# ============================================================
_model     = None
_tokenizer = None


def _load_model():
    """Downloads model from HuggingFace on first call, then caches."""
    global _model, _tokenizer

    if _model is not None:
        return  # already loaded

    print(f"Loading classifier from HuggingFace: {HF_MODEL_REPO} ...")

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)

    _model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_REPO,
        use_safetensors = True,
    ).to(_device)

    _model.eval()
    print("Classifier ready.\n")


# ============================================================
# PUBLIC API
# ============================================================
def predict(text: str) -> dict:
    """
    Classify a single mental health query.

    Parameters
    ----------
    text : str
        The user's natural language query.

    Returns
    -------
    dict with keys:
        label     (str)   Predicted class, e.g. "Suicidal"
        label_id  (int)   Numeric class index, e.g. 3
        scores    (dict)  Softmax probabilities for all four classes,
                          e.g. {"Normal": 0.02, "Anxiety": 0.05,
                                "Depression": 0.11, "Suicidal": 0.82}
    """
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        padding        = True,
        max_length     = MAX_LENGTH,
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs    = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    label_id = int(probs.argmax())

    return {
        "label"   : ID2LABEL[label_id],
        "label_id": label_id,
        "scores"  : {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


# ============================================================
# SMOKE TEST — run this file directly to verify the model works
# python src/classifier.py
# ============================================================
if __name__ == "__main__":
    samples = [
        ("I've been feeling really anxious and can't stop worrying.",  "Anxiety"),
        ("I don't see the point in living anymore.",                   "Suicidal"),
        ("I had a great day and feel very positive!",                  "Normal"),
        ("I feel completely empty and can't get out of bed.",          "Depression"),
    ]

    print("=" * 60)
    print("  CLASSIFIER SMOKE TEST")
    print("=" * 60)

    all_correct = True
    for text, expected in samples:
        result   = predict(text)
        correct  = result["label"] == expected
        status   = "✓" if correct else "✗"
        if not correct:
            all_correct = False
        print(f"\n  {status} Expected : {expected}")
        print(f"    Predicted: {result['label']} "
              f"({result['scores'][result['label']]:.1%} confidence)")
        print(f"    Text     : {text[:70]}")

    print("\n" + "=" * 60)
    print(f"  Result: {'All predictions correct ✓' if all_correct else 'Some predictions wrong ✗'}")
    print("=" * 60)