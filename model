"""
Product Review Sentiment Analysis
Dataset: Amazon Product Reviews (via Hugging Face datasets)
         - 'amazon_polarity' dataset: 4M+ reviews, binary sentiment (pos/neg)

Approach:
  1. Load & explore the dataset
  2. Preprocess text (clean, tokenize)
  3. Train a TF-IDF + Logistic Regression baseline
  4. Evaluate with classification report, confusion matrix, ROC curve
  5. Demonstrate inference on custom reviews
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_SIZE   = 50_000   # rows to use (full dataset is 4M – scale up freely)
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
MAX_FEATURES  = 50_000   # TF-IDF vocabulary cap
NGRAM_RANGE   = (1, 2)   # unigrams + bigrams

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 65)
print("  Product Review Sentiment Analysis")
print("  Dataset : amazon_polarity (Hugging Face)")
print("=" * 65)

print(f"\n[1/6] Loading dataset (sample={SAMPLE_SIZE:,}) …")
t0 = time.time()

ds = load_dataset("amazon_polarity", split="train")
df = ds.to_pandas().sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# Combine title + content for richer signal
df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")
df["sentiment"] = df["label"]   # 0 = negative, 1 = positive

print(f"   Loaded in {time.time()-t0:.1f}s  |  Shape: {df.shape}")
print(f"   Columns : {list(df.columns)}")

# ── 2. Explore ─────────────────────────────────────────────────────────────────
print("\n[2/6] Exploratory Data Analysis …")

label_counts = df["sentiment"].value_counts()
print(f"\n   Class distribution:\n{label_counts.rename({0:'Negative',1:'Positive'}).to_string()}")

avg_len = df["text"].str.split().str.len().describe()
print(f"\n   Review word-count stats:\n{avg_len.to_string()}")

# ── 3. Preprocess ──────────────────────────────────────────────────────────────
print("\n[3/6] Preprocessing text …")

def clean_text(text: str) -> str:
    """Lowercase, remove HTML tags, punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # HTML tags
    text = re.sub(r"[^a-z0-9\s']", " ", text)      # keep letters, digits, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
print("   Done.")

# ── 4. Train / Evaluate ────────────────────────────────────────────────────────
print("\n[4/6] Splitting and training …")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"],
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["sentiment"]
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=3,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

t0 = time.time()
pipeline.fit(X_train, y_train)
print(f"   Training time: {time.time()-t0:.1f}s")

# Predictions
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_proba)
print(f"\n   Accuracy : {acc:.4f}")
print(f"   ROC-AUC  : {auc:.4f}")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ── 5. Visualise ───────────────────────────────────────────────────────────────
print("[5/6] Generating visualisations …")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

DARK   = "#0f1117"
CARD   = "#1a1d27"
POS    = "#22d3a5"
NEG    = "#f43f5e"
ACCENT = "#818cf8"
TEXT   = "#e2e8f0"

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.facecolor": CARD, "figure.facecolor": DARK,
})

# — (a) Class distribution bar —
ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(["Negative", "Positive"], label_counts.values,
               color=[NEG, POS], width=0.5, edgecolor="none")
for bar, val in zip(bars, label_counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
             f"{val:,}", ha="center", fontsize=10, color=TEXT, fontweight="bold")
ax0.set_title("Class Distribution", color=TEXT, fontsize=13, pad=10)
ax0.set_ylabel("Count", color=TEXT)
ax0.spines[:].set_visible(False)
ax0.set_ylim(0, label_counts.max() * 1.15)

# — (b) Review length histogram —
ax1 = fig.add_subplot(gs[0, 1])
lengths = df["clean_text"].str.split().str.len()
ax1.hist(lengths.clip(upper=300), bins=50, color=ACCENT, alpha=0.85, edgecolor="none")
ax1.axvline(lengths.median(), color=POS, linewidth=1.5, linestyle="--",
            label=f"Median {int(lengths.median())} words")
ax1.set_title("Review Length Distribution", color=TEXT, fontsize=13, pad=10)
ax1.set_xlabel("Words per review")
ax1.set_ylabel("Frequency")
ax1.legend(fontsize=9)
ax1.spines[:].set_visible(False)

# — (c) ROC curve —
ax2 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color=POS, linewidth=2, label=f"AUC = {auc:.4f}")
ax2.plot([0,1],[0,1], color="#475569", linewidth=1, linestyle="--")
ax2.fill_between(fpr, tpr, alpha=0.15, color=POS)
ax2.set_title("ROC Curve", color=TEXT, fontsize=13, pad=10)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=10)
ax2.spines[:].set_visible(False)

# — (d) Confusion matrix —
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=",d", cmap="RdYlGn",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
            ax=ax3, linewidths=1, linecolor=DARK, cbar=False,
            annot_kws={"size": 13, "color": "#0f1117", "weight": "bold"})
ax3.set_title("Confusion Matrix", color=TEXT, fontsize=13, pad=10)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# — (e) Top positive features —
ax4 = fig.add_subplot(gs[1, 1])
feat_names = pipeline.named_steps["tfidf"].get_feature_names_out()
coef       = pipeline.named_steps["clf"].coef_[0]
top_n = 12
top_pos_idx = np.argsort(coef)[-top_n:][::-1]
top_pos_words = [feat_names[i] for i in top_pos_idx]
top_pos_vals  = coef[top_pos_idx]
ax4.barh(top_pos_words[::-1], top_pos_vals[::-1], color=POS, edgecolor="none")
ax4.set_title("Top Positive Features", color=TEXT, fontsize=13, pad=10)
ax4.set_xlabel("Logistic Regression Coefficient")
ax4.spines[:].set_visible(False)

# — (f) Top negative features —
ax5 = fig.add_subplot(gs[1, 2])
top_neg_idx = np.argsort(coef)[:top_n]
top_neg_words = [feat_names[i] for i in top_neg_idx]
top_neg_vals  = coef[top_neg_idx]
ax5.barh(top_neg_words[::-1], top_neg_vals[::-1], color=NEG, edgecolor="none")
ax5.set_title("Top Negative Features", color=TEXT, fontsize=13, pad=10)
ax5.set_xlabel("Logistic Regression Coefficient")
ax5.spines[:].set_visible(False)

fig.suptitle("Product Review Sentiment Analysis  ·  TF-IDF + Logistic Regression",
             fontsize=16, color=TEXT, fontweight="bold", y=0.98)

plt.savefig("/mnt/user-data/outputs/sentiment_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
print("   Saved → sentiment_dashboard.png")

# ── 6. Custom Inference ────────────────────────────────────────────────────────
print("\n[6/6] Custom review inference …")

custom_reviews = [
    "Absolutely love this product! Best purchase I've made all year. Works perfectly.",
    "Total waste of money. Broke after two days and customer support was useless.",
    "It's okay. Nothing special but does what it says. Shipping was fast.",
    "The quality is incredible, feels premium and looks even better in person!",
    "Arrived damaged, packaging was terrible. Would not recommend to anyone.",
]

print("\n   ┌─────────────────────────────────────────────────────────────────────")
print("   │  Custom Review Predictions")
print("   ├─────────────────────────────────────────────────────────────────────")
for review in custom_reviews:
    cleaned   = clean_text(review)
    pred      = pipeline.predict([cleaned])[0]
    prob      = pipeline.predict_proba([cleaned])[0]
    label     = "✅ POSITIVE" if pred == 1 else "❌ NEGATIVE"
    conf      = prob[pred]
    snippet   = review[:60] + ("…" if len(review) > 60 else "")
    print(f"   │  {label}  ({conf:.1%})  \"{snippet}\"")
print("   └─────────────────────────────────────────────────────────────────────")

print("\n✅  All done!")
print("   Outputs: sentiment_dashboard.png")
