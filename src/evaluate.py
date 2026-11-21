"""
evaluate.py  -  Stage 6 of the pipeline: score the models and compare them.

This file does NOT run any model. It only takes lists of text:
    predictions  = what a model wrote for each test clause
    references   = the human-written plain-English summaries
and returns numbers. The notebook calls it twice (base model, fine-tuned model)
and writes one comparison table to results/metrics.csv.

Metrics
    ROUGE-1 / ROUGE-2 / ROUGE-L : word overlap with the reference (higher = closer)
    BERTScore F1                : meaning similarity to the reference (higher = closer)
    Flesch Reading Ease         : how easy the output is to read (higher = easier)
    Avg words                   : average output length (helps explain the scores)

Run the self-test locally:   python src/evaluate.py
"""

import os
import random
import pandas as pd
from rouge_score import rouge_scorer
import textstat

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ----------------------------------------------------------------------------
# Individual metrics
# ----------------------------------------------------------------------------
def compute_rouge(predictions, references):
    """Average ROUGE-1, ROUGE-2 and ROUGE-L F1 over all pairs."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # note the order: (reference, prediction)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = len(predictions)
    return {key: round(value / n, 4) for key, value in totals.items()}


def compute_bertscore(predictions, references):
    """Average BERTScore F1. Imported inside the function because it needs
    PyTorch, which is only installed on Colab, not on your laptop."""
    from bert_score import score as bert_score
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
    return round(float(f1.mean()), 4)


def compute_flesch(texts):
    """Average Flesch Reading Ease (higher = easier). Empty outputs are skipped."""
    values = [textstat.flesch_reading_ease(t) for t in texts if str(t).strip()]
    return round(sum(values) / len(values), 2) if values else 0.0


def average_words(texts):
    return round(sum(len(str(t).split()) for t in texts) / len(texts), 1)


# ----------------------------------------------------------------------------
# One model -> one row of the results table
# ----------------------------------------------------------------------------
def evaluate_model(name, predictions, references, use_bertscore=True):
    assert len(predictions) == len(references), "predictions and references must match"
    row = {"model": name}
    rouge = compute_rouge(predictions, references)
    row["ROUGE-1"] = rouge["rouge1"]
    row["ROUGE-2"] = rouge["rouge2"]
    row["ROUGE-L"] = rouge["rougeL"]
    row["BERTScore-F1"] = compute_bertscore(predictions, references) if use_bertscore else None
    row["Flesch"] = compute_flesch(predictions)
    row["Avg words"] = average_words(predictions)
    return row


def build_comparison(rows):
    """Turn the list of rows (one per model) into one comparison table."""
    return pd.DataFrame(rows)


def save_metrics(df, filename="metrics.csv"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved metrics -> {os.path.normpath(path)}")
    return path


# ----------------------------------------------------------------------------
# Side-by-side examples for your error analysis
# ----------------------------------------------------------------------------
def save_examples(clauses, references, base_preds, ft_preds, n=8, seed=42,
                  filename="examples.md"):
    """Write n random test examples (same clause, reference, base output,
    fine-tuned output) to results/examples.md."""
    random.seed(seed)
    idx = random.sample(range(len(clauses)), min(n, len(clauses)))
    lines = ["# Example outputs: base vs fine-tuned\n"]
    for k, i in enumerate(idx, 1):
        lines += [
            f"## Example {k}\n",
            f"**Original clause:** {clauses[i]}\n",
            f"**Reference summary:** {references[i]}\n",
            f"**Base model:** {base_preds[i]}\n",
            f"**Fine-tuned model:** {ft_preds[i]}\n",
            "---\n",
        ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved examples -> {os.path.normpath(path)}")
    return path


# ----------------------------------------------------------------------------
# Self-test with dummy sentences (no model, no GPU, no BERTScore needed)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    refs = [
        "you may mod the game but don t distribute hacked clients.",
        "it comes with no warranty.",
    ]
    good = [
        "you can mod the game but you can t share hacked clients.",
        "there is no warranty.",
    ]
    bad = [
        "the licensee is hereby permitted to modify the software subject to restrictions on distribution.",
        "the provider expressly disclaims all warranties whether express or implied.",
    ]

    rows = [
        evaluate_model("good (dummy)", good, refs, use_bertscore=False),
        evaluate_model("bad (dummy)", bad, refs, use_bertscore=False),
    ]
    table = build_comparison(rows)
    print(table.to_string(index=False))
    print("\nExpected: the 'good' row has higher ROUGE and a higher Flesch score.")
