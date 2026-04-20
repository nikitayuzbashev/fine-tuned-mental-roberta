# This script was run on Google Colab (A100 GPU) to fine-tune MentalRoBERTa.
# It is NOT intended to run in the Streamlit environment.
# The trained model is published to HuggingFace Hub: nyuzbashev/mental-roberta-finetuned
"""
train_mental_roberta.py
=======================
Fine-tunes MentalRoBERTa (mental/mental-roberta-base) for four-class
mental health classification: Normal / Anxiety / Depression / Suicidal.

Designed to run on Google Colab with A100 GPU.

Setup (run once before this script)
-------------------------------------
1. Runtime > Change runtime type > A100 GPU
2. Upload Dataset_1_mental_heath_unbanlanced.csv to your Google Drive root
3. In a Colab cell, authenticate with HuggingFace to access the gated model:

       !pip install -q huggingface_hub
       from huggingface_hub import notebook_login
       notebook_login()

   Paste a READ token from https://huggingface.co/settings/tokens

Accuracy-maximising techniques used
-------------------------------------
- MentalRoBERTa: domain-specific RoBERTa pretrained on mental health corpora,
  giving a much stronger starting point than general RoBERTa.
- MAX_LENGTH = 512: maximum context.
- Gradient accumulation (x4): effective batch size of 64 improves gradient
  stability without requiring more GPU memory.
- Label smoothing (0.1): prevents overconfidence on noisy, Reddit-sourced
  labels where some posts are ambiguous or mislabelled.
- Weighted loss: corrects for class imbalance without distorting the real-world
  distribution via oversampling.
- Cosine LR schedule with warmup: smoother convergence than linear decay,
  consistently outperforms on fine-tuning tasks.
- Classifier dropout (0.2): reduces overfitting on the classification head.
- Early stopping on weighted-F1: halts when generalisation stops improving,
  avoids overfitting on majority classes.
- Suicidal recall tracked separately: primary safety metric, reported at every
  epoch. Misclassifying Suicidal as Normal is the costliest error.

Output
------
Trained model saved to YOUR_FOLDER_HERE/mental_roberta_finetuned/
Upload this folder to HuggingFace Hub after training completes.
The pre-trained model is available at: nyuzbashev/mental-roberta-finetuned
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Ensure script is not run - trained model is on HuggingFace
import sys
if "google.colab" not in sys.modules:
    raise SystemExit("train_mental_roberta.py is a Colab training script and should not be run here.")

# ============================================================
# 1. MOUNT GOOGLE DRIVE & RESOLVE PATHS
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DATASET_FILENAME = "Dataset_1_mental_heath_unbanlanced.csv"

# Update to your folder
DATASET_PATH = Path("/content/drive/MyDrive/YOUR_FOLDER_HERE") / DATASET_FILENAME

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find '{DATASET_FILENAME}'.\n"
        f"Expected at: {DATASET_PATH}\n"
        "Check your Drive path and re-run."
    )

# Model saved here: update with your folder structure
OUTPUT_DIR   = Path("/content/drive/MyDrive/YOUR_FOLDER_HERE/mental_roberta_finetuned")
RESULTS_DIR = Path("/content/results")
LOGS_DIR    = Path("/content/logs")

print(f"Dataset : {DATASET_PATH}")
print(f"Output  : {OUTPUT_DIR}\n")
# ============================================================
# 2. CONFIGURATION
# ============================================================
MODEL_NAME = "mental/mental-roberta-base"

LABEL_MAP = {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
ID2LABEL  = {v: k for k, v in LABEL_MAP.items()}

SEED = 42

# --- Accuracy-critical hyperparameters ---
MAX_LENGTH              = 512   # max
TRAIN_BATCH_SIZE        = 32    # fits comfortably on A100
GRAD_ACCUMULATION_STEPS = 2     # effective batch = 64, improves stability
EVAL_BATCH_SIZE         = 64
LEARNING_RATE           = 2e-5
WEIGHT_DECAY            = 0.01
WARMUP_RATIO            = 0.10
LABEL_SMOOTHING         = 0.1   # handles noisy Reddit labels
CLASSIFIER_DROPOUT      = 0.2   # reduces overfitting on classification head
MAX_EPOCHS              = 12    # hard ceiling; early stopping fires well before
EARLY_STOPPING_PATIENCE = 3
LOGGING_STEPS           = 100

# ============================================================
# 3. REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# 4. DEVICE
# ============================================================
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = device.type == "cuda"

print(f"\n{'='*55}")
print(f"  Device : {device}  |  fp16 : {use_fp16}")
print(f"{'='*55}\n")

if not torch.cuda.is_available():
    print("WARNING: No GPU detected. Training will be very slow.")
    print("Go to Runtime > Change runtime type > A100 GPU\n")

# ============================================================
# 5. DATA LOADING & SPLITTING  (70 / 15 / 15)
# ============================================================
print("Loading dataset ...")
df = pd.read_csv(str(DATASET_PATH))

df = df.dropna(subset=["text", "status"])
df["label"] = df["status"].map(LABEL_MAP)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"Total samples after cleaning : {len(df):,}")
for name, idx in LABEL_MAP.items():
    print(f"  {name:<12}: {(df['label'] == idx).sum():>6,}")

# Stratified 70/15/15 split
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(),
    test_size=0.30, random_state=SEED, stratify=df["label"],
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.50, random_state=SEED, stratify=temp_labels,
)

print(f"\nSplit -> Train: {len(train_texts):,} | "
      f"Val: {len(val_texts):,} | Test: {len(test_texts):,}\n")

# ============================================================
# 6. CLASS WEIGHTS
# ============================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:")
for name, idx in LABEL_MAP.items():
    print(f"  {name:<12}: {class_weights[idx]:.4f}")
print()

# ============================================================
# 7. TOKENISATION
# ============================================================
print("Loading tokeniser ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

print("Tokenising splits ...")
train_enc = tokenize(train_texts)
val_enc   = tokenize(val_texts)
test_enc  = tokenize(test_texts)

# ============================================================
# 8. PYTORCH DATASET
# ============================================================
class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = MentalHealthDataset(train_enc, train_labels)
val_dataset   = MentalHealthDataset(val_enc,   val_labels)
test_dataset  = MentalHealthDataset(test_enc,  test_labels)

# ============================================================
# 9. CUSTOM TRAINER
#    Combines class-weighted loss with label smoothing for noisy data
# ============================================================
class WeightedLabelSmoothingTrainer(Trainer):
    """
    CrossEntropyLoss with:
      - class weights: compensates for dataset imbalance
      - label smoothing: reduces overconfidence on noisy Reddit labels
    """
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fn = nn.CrossEntropyLoss(
            weight        = class_weights_tensor,
            label_smoothing = LABEL_SMOOTHING,
        )
        loss = loss_fn(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 10. METRICS
# ============================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # Suicidal recall tracked separately — primary safety metric
    _, per_class_recall, _, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
        labels=list(LABEL_MAP.values()),
    )
    suicidal_recall = per_class_recall[LABEL_MAP["Suicidal"]]

    return {
        "accuracy"        : acc,
        "f1"              : f1,
        "precision"       : precision,
        "recall"          : recall,
        "suicidal_recall" : suicidal_recall,
    }

# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir    = str(RESULTS_DIR),
    logging_dir   = str(LOGS_DIR),

    # Epochs & early stopping
    num_train_epochs              = MAX_EPOCHS,
    metric_for_best_model         = "f1",
    greater_is_better             = True,
    load_best_model_at_end        = True,
    eval_strategy                 = "epoch",
    save_strategy                 = "epoch",
    save_total_limit              = 2,

    # Batch & gradient accumulation
    per_device_train_batch_size   = TRAIN_BATCH_SIZE,
    per_device_eval_batch_size    = EVAL_BATCH_SIZE,
    gradient_accumulation_steps   = GRAD_ACCUMULATION_STEPS,

    # Optimiser
    learning_rate                 = LEARNING_RATE,
    weight_decay                  = WEIGHT_DECAY,
    warmup_ratio                  = WARMUP_RATIO,
    max_grad_norm                 = 1.0,
    lr_scheduler_type             = "cosine",  # smoother than linear decay

    # Mixed precision — A100 supports fp16
    fp16                          = use_fp16,

    # Logging
    logging_steps                 = LOGGING_STEPS,
    report_to                     = "none",

    # Reproducibility
    seed                          = SEED,
    data_seed                     = SEED,
)

# ============================================================
# 12. MODEL
#     Classifier dropout increased from default 0.1 to 0.2
#     to reduce overfitting on the classification head
# ============================================================
print("Loading MentalRoBERTa ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels              = len(LABEL_MAP),
    id2label                = ID2LABEL,
    label2id                = LABEL_MAP,
    ignore_mismatched_sizes = True,
    use_safetensors         = True,
    classifier_dropout      = CLASSIFIER_DROPOUT,
).to(device)

# ============================================================
# 13. TRAINER + EARLY STOPPING
# ============================================================
trainer = WeightedLabelSmoothingTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
    data_collator   = DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks       = [EarlyStoppingCallback(
                           early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

# ============================================================
# 14. TRAIN
# ============================================================
print(f"\nStarting training  (early stopping patience = "
      f"{EARLY_STOPPING_PATIENCE} epochs on weighted-F1) ...\n")
trainer.train()

# ============================================================
# 15. FINAL EVALUATION ON HELD-OUT 15% TEST SET
# ============================================================
print(f"\n{'='*55}")
print("  FINAL EVALUATION ON UNSEEN 15% TEST SET")
print(f"{'='*55}")

test_results = trainer.evaluate(test_dataset)

print(f"  Accuracy         : {test_results['eval_accuracy']        * 100:.2f} %")
print(f"  Weighted F1      : {test_results['eval_f1']              * 100:.2f} %")
print(f"  Weighted Prec.   : {test_results['eval_precision']       * 100:.2f} %")
print(f"  Weighted Recall  : {test_results['eval_recall']          * 100:.2f} %")
print(f"  Suicidal Recall  : {test_results['eval_suicidal_recall'] * 100:.2f} %")
print(f"{'='*55}\n")

# Detailed per-class breakdown
print("Per-class classification report:")
pred_output  = trainer.predict(test_dataset)
test_preds   = pred_output.predictions.argmax(-1)
target_names = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

print(classification_report(
    test_labels, test_preds,
    target_names=target_names, digits=4, zero_division=0,
))

print("Confusion matrix  (rows = actual, cols = predicted):")
cm = confusion_matrix(test_labels, test_preds)
print(f"{'':>12}" + "".join(f"{n:>12}" for n in target_names))
for i, row in enumerate(cm):
    print(f"{target_names[i]:>12}" + "".join(f"{v:>12}" for v in row))

# ============================================================
# 16. SAVE TO GOOGLE DRIVE
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"\nModel saved to Google Drive at '{OUTPUT_DIR}'")
