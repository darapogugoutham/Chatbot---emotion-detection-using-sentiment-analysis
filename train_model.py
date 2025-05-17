import os
import re
import warnings
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

# ====== SETUP ======
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ====== Load Sentiment140 train and test splits ======
print("Loading Sentiment140 train and test splits...")
dataset = load_dataset("sentiment140", trust_remote_code=True)
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])
df = pd.concat([df_train, df_test], ignore_index=True)[["text", "sentiment"]]

# ====== Map sentiments ======
label_mapping = {0: 0, 2: 1, 4: 2}  # 0=neg, 2=neutral, 4=pos → 0,1,2
df = df[df["sentiment"].isin(label_mapping.keys())]
df["label"] = df["sentiment"].map(label_mapping)
df = df.drop(columns=["sentiment"])

# ====== Sample balanced data ======
df_neg = df[df["label"] == 0].sample(n=7000, random_state=42)
df_pos = df[df["label"] == 2].sample(n=7000, random_state=42)
df_neu = df[df["label"] == 1]  # all neutral
print(f"Neutral samples found: {len(df_neu)}")

df = pd.concat([df_neg, df_neu, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("final_balanced_with_neutral.csv", index=False)
print(f"Saved combined dataset to final_balanced_with_neutral.csv with shape: {df.shape}")

# ====== Preprocessing ======
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# ====== Tokenization ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# ====== Train/Validation/Test Split ======
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42)

train_label_counts = train_df["label"].value_counts().sort_index()
val_label_counts = val_df["label"].value_counts().sort_index()
test_label_counts = test_df["label"].value_counts().sort_index()

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# ====== Metrics ======
def compute_metrics(pred):
    labels_pred = np.argmax(pred.predictions, axis=1)
    labels_true = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average='weighted', zero_division=0)
    acc = accuracy_score(labels_true, labels_pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ====== Load BERT Model ======
emotion_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2}
)

# ====== Training Arguments ======
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=500,
    logging_steps=100,
    no_cuda=not torch.cuda.is_available()
)

# ====== Trainer ======
trainer = Trainer(
    model=emotion_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ====== Train Model ======
print("Starting training...")
trainer.train()

# ====== Save Model ======
emotion_model.save_pretrained("./fine_tuned_sentiment_model")
tokenizer.save_pretrained("./fine_tuned_sentiment_model")
print("Fine-tuning complete and model saved")

# ====== Evaluate Model ======
train_predictions = trainer.predict(train_dataset)
val_predictions = trainer.predict(val_dataset)
test_predictions = trainer.predict(test_dataset)

train_preds = np.argmax(train_predictions.predictions, axis=1)
val_preds = np.argmax(val_predictions.predictions, axis=1)
test_preds = np.argmax(test_predictions.predictions, axis=1)

train_labels_true = train_predictions.label_ids
val_labels_true = val_predictions.label_ids
test_labels_true = test_predictions.label_ids

# ====== Report ======
labels_out = ["negative", "neutral", "positive"]

with open("training_report.txt", "w") as f:
    f.write("Class Labels:\n")
    f.write(f"{labels_out}\n\n")

    f.write("Dataset Sizes:\n")
    f.write(f"Training samples: {len(train_dataset)}\n")
    f.write(f"Validation samples: {len(val_dataset)}\n")
    f.write(f"Testing samples: {len(test_dataset)}\n\n")

    f.write("Label Distribution (Train):\n")
    f.write(f"{train_label_counts.to_string()}\n\n")
    f.write("Label Distribution (Validation):\n")
    f.write(f"{val_label_counts.to_string()}\n\n")
    f.write("Label Distribution (Test):\n")
    f.write(f"{test_label_counts.to_string()}\n\n")

    f.write("\n====== Train Set Evaluation ======\n")
    f.write(classification_report(train_labels_true, train_preds, labels=[0, 1, 2], target_names=labels_out, zero_division=0))
    f.write("\nConfusion Matrix (Train):\n")
    f.write(np.array2string(confusion_matrix(train_labels_true, train_preds, labels=[0, 1, 2])))

    f.write("\n\n====== Validation Set Evaluation ======\n")
    f.write(classification_report(val_labels_true, val_preds, labels=[0, 1, 2], target_names=labels_out, zero_division=0))
    f.write("\nConfusion Matrix (Validation):\n")
    f.write(np.array2string(confusion_matrix(val_labels_true, val_preds, labels=[0, 1, 2])))

    f.write("\n\n====== Test Set Evaluation ======\n")
    f.write(classification_report(test_labels_true, test_preds, labels=[0, 1, 2], target_names=labels_out, zero_division=0))
    f.write("\nConfusion Matrix (Test):\n")
    f.write(np.array2string(confusion_matrix(test_labels_true, test_preds, labels=[0, 1, 2])))

print("Training report saved to training_report.txt")
