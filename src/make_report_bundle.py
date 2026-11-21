"""
make_report_bundle.py - packs all results into ONE file for the web report.

Run it after the notebook has produced these files in results/:
    metrics.csv, base_outputs.json, ft_outputs.json, train_log.json
It writes results/results_bundle.json, which you drop into web/index.html
(or publish next to it so the page loads it automatically).

    python src/make_report_bundle.py
"""
import os, json
import pandas as pd
from rouge_score import rouge_scorer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "processed", "test.jsonl")


def load_json(name, required=True):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"Missing results/{name}. Run the notebook first and copy results/ from Drive.")
        return None
    return json.load(open(path, encoding="utf-8"))


def main():
    metrics = pd.read_csv(os.path.join(RES, "metrics.csv")).to_dict(orient="records")
    base = load_json("base_outputs.json")
    ft = load_json("ft_outputs.json")
    log = load_json("train_log.json", required=False) or []
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    assert len(base) == len(ft) == len(test), "output counts must match the test set"

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    examples = []
    for t, b, f in zip(test, base, ft):
        examples.append({
            "clause": t["clause"], "reference": t["target"], "base": b, "ft": f,
            "base_rl": round(sc.score(t["target"], b)["rougeL"].fmeasure, 4),
            "ft_rl": round(sc.score(t["target"], f)["rougeL"].fmeasure, 4),
        })

    bundle = {
        "meta": {
            "title": "Legal Text Simplification with LoRA",
            "model": "Qwen2.5-3B-Instruct",
            "method": "LoRA / QLoRA 4-bit",
            "dataset": "joelniklaus/plain_english_contracts_summarization",
            "n_test": len(test),
        },
        "hyper": {"LoRA rank (r)": 16, "LoRA alpha": 32, "Dropout": 0.05, "Learning rate": "2e-4",
                  "Epochs": 3, "Batch size": "4 x 4 accumulation = 16", "Max length": 512, "Seed": 42},
        "metrics": metrics,
        "examples": examples,
        "loss": {
            "train": [[l["step"], l["loss"]] for l in log if "loss" in l],
            "eval": [[l["step"], l["eval_loss"]] for l in log if "eval_loss" in l],
        },
    }
    out = os.path.join(RES, "results_bundle.json")
    json.dump(bundle, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("Saved", os.path.normpath(out))


if __name__ == "__main__":
    main()
