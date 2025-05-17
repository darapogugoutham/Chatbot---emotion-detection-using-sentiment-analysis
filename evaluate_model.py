import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import re

# ====== Load model and tokenizer ======
model_path = "./fine_tuned_sentiment_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# ====== Load and preprocess datasets ======
df = pd.read_csv("final_balanced_with_neutral.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# ====== Train/Val/Test Split ======
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42)

# ====== Tokenize ======
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# ====== Predict and Evaluate ======
trainer = Trainer(model=model)

def save_metrics_table_as_image(acc, precision, recall, f1, dataset_name):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    col_labels = ["Metric", "Score"]
    cell_text = [
        ["Accuracy", f"{acc:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"]
    ]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title(f"{dataset_name} Set Evaluation Metrics", fontsize=14)
    plt.tight_layout()
    os.makedirs("evaluation_figures", exist_ok=True)
    plt.savefig(f"evaluation_figures/evaluation_metrics_table_{dataset_name.lower()}.jpg")
    plt.close()

def evaluate(dataset, dataset_name):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    print(f"\n==== {dataset_name.upper()} SET ====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(labels, preds, labels=[0, 1, 2], target_names=["negative", "neutral", "positive"]))

    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"{dataset_name} Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"evaluation_figures/confusion_matrix_{dataset_name.lower()}.jpg")
    plt.close()

    # Save table image
    save_metrics_table_as_image(acc, precision, recall, f1, dataset_name)

    return acc, precision, recall, f1, cm

# ====== Run Evaluation ======
evaluate(train_dataset, "Train")
evaluate(val_dataset, "Validation")
evaluate(test_dataset, "Test")

print("\nEvaluation figures saved in: evaluation_figures/")
