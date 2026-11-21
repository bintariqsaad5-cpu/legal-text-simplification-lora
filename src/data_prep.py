"""
data_prep.py  -  Stage 1 of the pipeline: get the raw dataset ready for the model.

What this file does, in order:
  1. Load the dataset from Hugging Face (it only has a 'train' split, 446 rows).
  2. Keep only the two columns we need: the legal clause and its plain-English summary.
  3. Clean the data (remove empty, duplicate, too-short and too-long rows).
  4. Split it ourselves into train / validation / test with a fixed seed.
  5. Build the PROMPT that every other file will reuse (training, baseline, demo).
  6. Save the three splits as .jsonl files so the test set stays fixed.

Run it from the project root:   python src/data_prep.py
"""

import os
import pandas as pd
from datasets import load_dataset

# ----------------------------------------------------------------------------
# Settings (change these here, nowhere else)
# ----------------------------------------------------------------------------
DATASET_ID = "joelniklaus/plain_english_contracts_summarization"
TEXT_COL = "original_text"        # the legal clause (input)
SUMMARY_COL = "reference_summary"  # the plain-English summary (target)

SEED = 42                  # fixed seed so the split is the same every time
TRAIN_FRAC = 0.70          # 70% train
VAL_FRAC = 0.15            # 15% validation, the remaining 15% is test

# Cleaning rules. The first run prints how many rows each rule removes,
# so you can tune these numbers after looking at the statistics.
MIN_SUMMARY_WORDS = 5      # drops junk summaries like "hi."
MIN_CLAUSE_WORDS = 10
MAX_CLAUSE_WORDS = 250     # very long clauses use too much GPU memory

# Where the cleaned splits are saved (data/processed/ in the project root)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

# ----------------------------------------------------------------------------
# The prompt template. This is the ONE place it is defined.
# Training, the base-model baseline and the Gradio demo must all use it,
# otherwise the base vs fine-tuned comparison is not fair.
# ----------------------------------------------------------------------------
INSTRUCTION = "Rewrite the following legal clause in plain English.\n\nClause: "


def build_prompt(clause):
    """Turn one legal clause into the text we send to the model."""
    return INSTRUCTION + clause.strip()


# ----------------------------------------------------------------------------
# Step 1 and 2: load and keep only the columns we need
# ----------------------------------------------------------------------------
def load_raw():
    ds = load_dataset(DATASET_ID)["train"]
    df = ds.to_pandas()[[TEXT_COL, SUMMARY_COL]]
    df = df.rename(columns={TEXT_COL: "clause", SUMMARY_COL: "summary"})
    return df


def word_count(text):
    return len(str(text).split())


# ----------------------------------------------------------------------------
# Step 3: cleaning. Each rule prints how many rows it removed.
# ----------------------------------------------------------------------------
def clean(df):
    print(f"Rows before cleaning: {len(df)}")

    # remove missing values and strip extra spaces
    df = df.dropna().copy()
    df["clause"] = df["clause"].astype(str).str.strip()
    df["summary"] = df["summary"].astype(str).str.strip()
    n = len(df)
    df = df[(df["clause"] != "") & (df["summary"] != "")]
    print(f"  removed empty rows:            {n - len(df)}")

    # remove duplicate clauses (a repeated clause could leak into the test set)
    n = len(df)
    df = df.drop_duplicates(subset="clause")
    print(f"  removed duplicate clauses:     {n - len(df)}")

    df["clause_words"] = df["clause"].apply(word_count)
    df["summary_words"] = df["summary"].apply(word_count)

    # junk summaries such as "hi."
    n = len(df)
    df = df[df["summary_words"] >= MIN_SUMMARY_WORDS]
    print(f"  removed too-short summaries:   {n - len(df)}")

    # clauses that are too short or too long
    n = len(df)
    df = df[(df["clause_words"] >= MIN_CLAUSE_WORDS) & (df["clause_words"] <= MAX_CLAUSE_WORDS)]
    print(f"  removed bad-length clauses:    {n - len(df)}")

    # a "simplification" longer than the original is not a summary
    n = len(df)
    df = df[df["summary_words"] < df["clause_words"]]
    print(f"  removed summary >= clause:     {n - len(df)}")

    df = df.reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")
    return df


# ----------------------------------------------------------------------------
# Step 4 and 5: split, then add the prompt and target columns
# ----------------------------------------------------------------------------
def split(df):
    # shuffle once with the fixed seed, then cut into three parts
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(len(df) * TRAIN_FRAC)
    n_val = int(len(df) * VAL_FRAC)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()   # the test set is never used in training
    return train, val, test


def add_prompt_columns(df):
    df = df.copy()
    df["prompt"] = df["clause"].apply(build_prompt)
    df["target"] = df["summary"]
    return df


# ----------------------------------------------------------------------------
# Step 6: save
# ----------------------------------------------------------------------------
def save_splits(train, val, test):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        part.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"  saved {len(part):4d} rows -> {os.path.normpath(path)}")


def load_splits():
    """Used by the notebook: read the saved splits back."""
    return tuple(
        pd.read_json(os.path.join(OUT_DIR, f"{name}.jsonl"), lines=True)
        for name in ("train", "val", "test")
    )


# ----------------------------------------------------------------------------
# Run the whole pipeline when you execute this file directly
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw()

    print("\n--- Word-count statistics BEFORE cleaning ---")
    stats = pd.DataFrame({
        "clause_words": raw["clause"].apply(word_count),
        "summary_words": raw["summary"].apply(word_count),
    })
    print(stats.describe(percentiles=[0.05, 0.5, 0.95]).round(1))

    print("\n--- Cleaning ---")
    cleaned = clean(raw)

    train, val, test = split(cleaned)
    train, val, test = (add_prompt_columns(p) for p in (train, val, test))

    print("\n--- Split sizes ---")
    print(f"train: {len(train)}   val: {len(val)}   test: {len(test)}   total: {len(train) + len(val) + len(test)}")

    print("\n--- Saving ---")
    save_splits(train, val, test)

    print("\n--- Three training examples ---")
    for i in range(3):
        row = train.iloc[i]
        print(f"\n[Example {i + 1}]")
        print("PROMPT:\n" + row["prompt"])
        print("TARGET:\n" + row["target"])
