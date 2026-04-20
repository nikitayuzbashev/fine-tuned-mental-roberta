# MentalRoBERTa — Fine-tuned Mental Health Classifier

A fine-tuned [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base) model for four-class mental health text classification, achieving **88.11% accuracy** and **88.17% weighted F1** on a held-out test set of 7,442 Reddit posts.

The fine-tuned model is publicly available on HuggingFace Hub:
**[nyuzbashev/mental-roberta-finetuned](https://huggingface.co/nyuzbashev/mental-roberta-finetuned)**

---

## What It Does

Given a natural language mental health query, the classifier assigns one of four labels:

| Label | Description |
|---|---|
| `Normal` | No significant mental health concern detected |
| `Anxiety` | Anxiety-related distress signals |
| `Depression` | Depressive episode indicators |
| `Suicidal` | Suicidal ideation or crisis signals |

This classifier was built as the first stage of a hybrid RAG (Retrieval-Augmented Generation) pipeline for a mental health crisis support system, developed as part of an MSc Information Retrieval project at Queen Mary University of London. The classifier routes queries to the appropriate verified clinical resources, ensuring responses are grounded in authoritative UK sources (NHS, Mind, Samaritans) rather than generated freely by an LLM.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 88.11% |
| Weighted F1 | 88.17% |
| Weighted Precision | 88.41% |
| Weighted Recall | 88.11% |
| **Suicidal Recall** | **86.03%** |

Suicidal Recall is the primary safety metric — it measures how many genuine crisis posts the model correctly identifies. Misclassifying a Suicidal post as Normal would be a critical failure in a clinical context, so this metric is tracked separately throughout training.

Per-class breakdown:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.9749 | 0.9565 | 0.9656 |
| Anxiety | 0.9196 | 0.9152 | 0.9174 |
| Depression | 0.8449 | 0.7886 | 0.8158 |
| Suicidal | 0.7685 | 0.8603 | 0.8118 |

For full results including the training curve, confusion matrix, and analysis, see [`results/evaluation_report.txt`](results/evaluation_report.txt).

---

## Why 88% Is Strong For This Task

Published academic results on similar Reddit-sourced 4-class mental health classification datasets typically report 85–92% weighted F1. The main challenge is the semantic overlap between Depression and Suicidal posts — both classes use overlapping vocabulary ("I feel hopeless", "nothing matters") and even human annotators struggle to distinguish them from text alone. The kNN + TF-IDF baseline on the same dataset achieved ~72% accuracy, demonstrating the substantial gain from domain-specific transformer fine-tuning.

---

## Project Structure

```
mental-roberta-classifier/
├── data/                          # Dataset directory (CSV gitignored — see below)
│   └── .gitkeep
├── notebooks/
│   └── mental_roberta_training.ipynb   # Original Colab training run with outputs
├── results/
│   └── evaluation_report.txt     # Full evaluation metrics and analysis
├── scripts/
│   └── train_mental_roberta.py   # Training script (Colab/GPU only)
├── src/
│   └── classifier.py             # Inference module — import this in your project
├── tests/
│   └── test_classifier.py        # Pytest test suite
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start — Using The Classifier

The fastest way to use this classifier is to pull the model directly from HuggingFace. No training required.

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Use in your code:**

```python
from src.classifier import predict

result = predict("I've been feeling really anxious and can't stop worrying.")

print(result["label"])     # "Anxiety"
print(result["label_id"])  # 1
print(result["scores"])    # {"Normal": 0.02, "Anxiety": 0.97, "Depression": 0.01, "Suicidal": 0.00}
```

The model downloads from HuggingFace on first use (~500MB) and is cached locally at `~/.cache/huggingface/`. Subsequent calls load from cache instantly.

**Verify the model works:**

```bash
python src/classifier.py
```

This runs the built-in smoke test — four sample predictions with expected labels printed to the terminal.

---

## Dataset

The model was trained on the **Mental Health Text Classification Dataset** from Kaggle:
[kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

Download the CSV and place it at `data/Dataset_1_mental_heath_unbanlanced.csv` before running training. The file is gitignored because it is not ours to redistribute.

Dataset statistics after cleaning:

| Class | Samples | % |
|---|---|---|
| Normal | 18,391 | 37.1% |
| Depression | 14,506 | 29.2% |
| Suicidal | 11,212 | 22.6% |
| Anxiety | 5,503 | 11.1% |
| **Total** | **49,612** | |

---

## Reproducing The Training

Training was run on **Google Colab with an A100 GPU** and takes approximately 30–45 minutes.

**Step 1 — Set up Colab:**
Go to Runtime → Change runtime type → A100 GPU.

**Step 2 — Install dependencies in a Colab cell:**
```python
!pip install -q torch transformers accelerate scikit-learn pandas numpy huggingface_hub
```

**Step 3 — Authenticate with HuggingFace** (required to access the gated base model):
```python
from huggingface_hub import notebook_login
notebook_login()
```
Paste a READ token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Step 4 — Update the path placeholders** in `scripts/train_mental_roberta.py`:
```python
DATASET_PATH = Path("/content/drive/MyDrive/YOUR_FOLDER_HERE") / DATASET_FILENAME
OUTPUT_DIR   = Path("/content/drive/MyDrive/YOUR_FOLDER_HERE/mental_roberta_finetuned")
```

**Step 5 — Run the script** in a Colab cell:
```python
exec(open("train_mental_roberta.py").read())
```

The trained model will be saved to your Google Drive. You can then upload it to HuggingFace Hub manually via the web UI or using the HuggingFace CLI.

Alternatively, see [`notebooks/mental_roberta_training.ipynb`](notebooks/mental_roberta_training.ipynb) for the exact Colab notebook used to produce the published model.

---

## Key Engineering Decisions

**Why MentalRoBERTa and not standard RoBERTa?** MentalRoBERTa was pretrained specifically on mental health corpora, giving it a much stronger semantic foundation for this domain than a general-purpose RoBERTa. The domain-specific vocabulary — clinical terminology, colloquial expressions of distress, Reddit-specific language — is already partially encoded in the base model's weights before fine-tuning begins.

**Why MAX_LENGTH=512?** A data analysis of the training corpus showed that 98.8% of posts fit within 512 tokens, which is RoBERTa's hard maximum. Using 128 (a common default) would truncate 28.4% of posts mid-sentence, discarding valuable context. Using 512 captures nearly the full emotional arc of each post.

**Why weighted loss + label smoothing together?** These two techniques solve different problems. Class weights fix the imbalance between Normal (18k samples) and Anxiety (5.5k samples) — without them, the model would learn to over-predict the majority class. Label smoothing addresses the noise within each class — Reddit posts are often ambiguous or sarcastically labelled, and smoothing prevents the model from becoming overconfident on labels that may be wrong. They are complementary rather than redundant.

**Why early stopping on F1 rather than loss?** Validation loss can continue decreasing even as per-class F1 plateaus or degrades, particularly when class imbalance is present. F1 directly measures what we care about — correct classification across all four classes — so it is a more reliable stopping criterion than loss for this task.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests use mocking to run without downloading the model, so the full test suite completes in under 5 seconds. See [`tests/test_classifier.py`](tests/test_classifier.py) for details.

---

## Environment Setup

```bash
# Create and activate a virtual environment (Python 3.11 recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your HuggingFace token if using a private model
```

---

## Acknowledgements

Base model: [mental/mental-roberta-base](https://huggingface.co/mental/mental-roberta-base)
Dataset: [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
Built as part of ECS7005P Information Retrieval, Queen Mary University of London, 2026.

---

## Disclaimer

This model is intended for research and educational purposes only. It is not a clinical diagnostic tool and must not be used as a substitute for professional mental health assessment. The labels produced by this classifier are not clinical diagnoses.
