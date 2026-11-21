# Legal Text Simplification with a LoRA Fine-Tuned Small Language Model

NLP Course Project | Category: Fine-Tuning Small Language Models (SLMs)

**Group members:** Saad Bin Tariq
**Instructor / Course:** Maryam Nasim, NLP, 7th Semester

---

## 1. Project Overview

Legal contracts and terms of service are hard for ordinary people to read. This project fine-tunes a small pretrained language model (3 billion parameters, Qwen2.5-3B-Instruct) using LoRA so that it rewrites a legal clause as a short, plain-English summary.

**Main question:** Does fine-tuning actually improve the model, compared with the same model used without fine-tuning (the base model), when both are tested on a held-out test set?

## 2. Dataset

- **Name:** Plain English Summarization of Contracts (Manor & Li, 2019)
- **Source:** Hugging Face, `joelniklaus/plain_english_contracts_summarization`
- **Content:** Legal text snippets from terms of service paired with plain-English summaries.
- **Size:** Under 1,000 examples (small, so results should be read with care).
- **Split:** Train / Validation / Test = 70 / 15 / 15, with a fixed random seed. The test set is never used during training.
- **Preprocessing:** Removed empty and duplicate rows, dropped very long examples, and formatted each pair into an instruction prompt.

> Check the dataset page for its licence and column names before use.

## 3. Model and Method

- **Base model:** Qwen2.5-3B-Instruct
- **Fine-tuning method:** LoRA / QLoRA (4-bit) using Hugging Face PEFT and TRL
- **Hardware:** Free Google Colab / Kaggle T4 GPU
- **Prompt format:** "Rewrite the following legal clause in plain English: ..." followed by the target summary.

| Setting | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 (dropout 0.05) |
| Learning rate | 2e-4 (cosine) |
| Epochs | 3 |
| Batch size | 4 x 4 accumulation = 16 |
| Random seed | 42 |

## 4. Evaluation

Both the base model and the fine-tuned model generate outputs for the **same held-out test clauses** using the **same prompt**.

| Metric | What it measures |
|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | Word overlap with the reference summary |
| BERTScore | Meaning similarity to the reference summary |
| Flesch Reading Ease | How easy the output is to read (higher is easier) |

## 5. Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Flesch Reading Ease |
|---|---|---|---|---|---|
| Base model | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned model | TBD | TBD | TBD | TBD | TBD |

Training loss curve: see `results/loss_curve.png`
Full numbers: see `results/metrics.csv`

## 6. Example Outputs

Side-by-side examples are in `results/examples.md`.

| Original clause | Reference summary | Base model output | Fine-tuned output |
|---|---|---|---|
| to fill in | to fill in | to fill in | to fill in |

## 7. Error Analysis and Limitations

To be completed after evaluation. Points to cover:

- Where fine-tuning helped and where it did not.
- Dropped clauses or missing conditions in the simplified text.
- Hallucinated or changed meaning.
- Small dataset size and its effect on reliability.
- Simplified text is **not legal advice** and should not replace reading the original contract.

## 8. Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── legal_simplification_lora.ipynb   Main notebook (data, baseline, training)
├── src/
│   ├── data_prep.py                      Cleaning, splitting, prompt formatting
│   └── evaluate.py                       ROUGE, BERTScore, Flesch scoring
├── web/
│   └── index.html                        Results report page (drop results_bundle.json)
├── app/
│   └── app.py                            Gradio demo
└── results/
    ├── metrics.csv
    ├── examples.md
    └── loss_curve.png
```

## 9. How to Run

1. Open `notebooks/legal_simplification_lora.ipynb` in Google Colab (File > Open notebook > GitHub).
2. Set the runtime to **T4 GPU** (Runtime > Change runtime type).
3. Run the install cell, then run all cells in order.
4. The trained LoRA adapter is saved to Google Drive (or Hugging Face Hub).
5. To launch the demo, run `app/app.py` and open the Gradio link.

For local setup (data preparation and evaluation only, no GPU needed):

1. Create and activate a virtual environment.
2. Install the libraries from `requirements.txt`.

## 10. Tools and Technologies

Python, Hugging Face Transformers, Datasets, PEFT, TRL, BitsAndBytes, Evaluate, BERTScore, Textstat, Gradio, Google Colab / Kaggle, Git and GitHub, Visual Studio Code.

## 11. References

- Manor, L. and Li, J. J. (2019). *Plain English Summarization of Contracts.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*