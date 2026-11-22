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
- **Fine-tuning method:** QLoRA (4-bit base model, LoRA adapters) using Hugging Face Transformers and PEFT. Loss is computed only on the summary tokens; the prompt tokens are masked.
- **Trainable parameters:** 29,933,568 of 3,115,872,256 (0.96%)
- **Hardware:** Free Google Colab / Kaggle T4 GPU
- **Prompt format:** "Rewrite the following legal clause in plain English: ..." followed by the target summary.

| Setting | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Learning rate | 2e-4 (cosine schedule, 3 warmup steps) |
| Epochs | 3 |
| Batch size | 4 per step x 4 accumulation = 16 effective |
| Max sequence length | 512 tokens |
| Precision | fp16 compute, 4-bit NF4 base weights (the T4 GPU has no bfloat16) |
| Decoding | Greedy, max 100 new tokens |
| Random seed | 42 |
| Data split | 273 train / 58 validation / 60 test (391 rows after cleaning 446) |

## 4. Evaluation

Both the base model and the fine-tuned model generate outputs for the **same held-out test clauses** using the **same prompt**.

| Metric | What it measures |
|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | Word overlap with the reference summary |
| BERTScore | Meaning similarity to the reference summary |
| Flesch Reading Ease | How easy the output is to read (higher is easier) |

## 5. Results

Evaluated on the same 60 held-out test clauses, same prompt, greedy decoding.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Flesch Reading Ease | Avg. words |
|---|---|---|---|---|---|---|
| Base Qwen2.5-3B-Instruct | 0.1884 | 0.0337 | 0.1286 | 0.8553 | 62.24 | 55.3 |
| Fine-tuned (LoRA) | 0.3896 | 0.2136 | 0.3494 | 0.9023 | 60.27 | 12.1 |

- The fine-tuned model scored higher ROUGE-L than the base model on **48 of 60** test clauses.
- Training took about 6.5 minutes (54 steps) on a free T4 GPU. Training loss fell from 2.02 to 1.19 over 3 epochs. Validation loss fell from 1.90 to 1.80 and flattened after epoch 2 (1.801 to 1.796), so more epochs would mostly overfit.
- **How to read these numbers:** the reference summaries average 17 words on the test set (median 13 over the whole dataset). The fine-tuned model writes about 12 words, while the base model writes about 55 and often adds a preamble such as "Here's the legal clause rewritten in simpler language" (30 of 60 base outputs start this way). Part of the ROUGE gain therefore comes from learning the short style of the references, not only from better understanding. Flesch Reading Ease did not improve (62.24 vs 60.27).
- Base outputs were capped at 100 new tokens, and 14 of 60 do not end with a full sentence (a simple punctuation check), so the cap hurts the base model's scores somewhat.

Training loss curve: see `results/loss_curve.png`
Full numbers: see `results/metrics.csv`. The ROUGE scores and the 48-of-60 count were recomputed from the saved outputs in `results/` and match the notebook.

## 6. Example Outputs

Side-by-side examples are in `results/examples.md`.

| Clause (shortened) | Reference summary | Base model | Fine-tuned |
|---|---|---|---|
| Spotify disclaims warranties of satisfactory quality, merchantability, fitness... | there is no guarantee of quality of service. if the service is not delivered to your expectations you still pay for it... | Long, mostly correct explanation, but adds details about viruses | spotify makes no guarantees regarding quality fitness for a particular purpose or non infringement. |
| Static page with no scripts, cookies, ads or trackers... | ip addresses of website visitors are not tracked. | Long, friendly paraphrase (cut off at the token limit) | this service does not track users. |
| Facebook may use your name and profile picture next to ads without compensation | your identity is used in ads that are shown to other users. | Accurate paraphrase of the clause | facebook can use your personal data for any reason they see fit. |

### Live demo results (Gradio app, screenshots in `results/app_results/`)

Four new clauses (not from the dataset) were pasted into the running app.

| Clause | Base model | Fine-tuned model | Comment |
|---|---|---|---|
| "As is" warranty disclaimer | Long but accurate | "the service is provided as is with no warranty." | Fine-tuned output is short and acceptable |
| Account suspension if the user breaks the rules or commits fraud | Accurate, keeps the conditions | "the service can delete your account for any reason they see fit." | Fine-tuned model drops the conditions and changes the meaning |
| Limitation of liability for indirect damages | Long and accurate | "the court of law limits liability for damages." | Fine-tuned output is vague and misleading |
| Changing the terms, with notice, and acceptance by continued use | Accurate, keeps notice and acceptance | "the service can change its terms any time they see fit." | Fine-tuned model drops notice and acceptance |

## 7. Error Analysis and Limitations

**Where fine-tuning helped.** The fine-tuned model learned the target format: short, direct summaries close to the references. This raised ROUGE-1 from 0.19 to 0.39, ROUGE-L from 0.13 to 0.35 and BERTScore F1 from 0.855 to 0.902, and it removed the long preambles of the base model. On short, simple clauses (for example the no-tracking page) it gave a correct one-line summary.

**Where it failed.** Higher overlap with the reference did not mean a more faithful summary.
- **Dropped conditions.** For a clause allowing account suspension only if the user breaks the rules or commits fraud, the fine-tuned model wrote "the service can delete your account for any reason they see fit", which removes the condition. The base model kept it.
- **Dropped consequences.** For a clause saying continued use means you accept changed terms, the fine-tuned output kept only "the service can change its terms any time they see fit".
- **Wrong or invented content.** An Apple terms-of-use clause was summarized as being about account deletion, and an Apple privacy clause as being about marketing.
- **Repeated template phrases.** The phrase "see fit" appears in 6 of the 60 fine-tuned test outputs and in the demo, across unrelated clauses, which suggests the model copies frequent patterns from the small training set.

**Data limitations.** The dataset is small (273 training examples) and noisy. Some reference summaries do not match their clause (one clause about content licensing is paired with a summary about data retention, and another says 18 years old while the reference says 13), which lowers scores unfairly and teaches the model wrong patterns.

**Evaluation limitations.**
- The test set has only 60 clauses, from a single run and a single random seed.
- ROUGE rewards matching short references, so it favors short outputs. The "wins on 48 of 60 clauses" count uses ROUGE-L only and says nothing about faithfulness.
- The base model got one plain prompt and a 100-token cap. A prompt that asks for one short sentence, or a few examples in the prompt, might narrow the gap.
- Flesch Reading Ease is unreliable on one-line outputs.
- The model was trained on clauses of 250 words or fewer. It is not reliable on longer or heavily nested text.

**Conclusion.** Fine-tuning clearly improves similarity to the reference summaries, but it does not make the summaries more accurate. For legal text, where a missing condition changes the meaning, the output should be checked by a person.

The simplified text is **not legal advice** and should not replace reading the original contract.

## 8. Repository Structure

```
legal-text-simplification-lora/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── legal_simplification_lora.ipynb   Main notebook, all outputs saved
├── src/
│   ├── data_prep.py                      Cleaning, splitting, prompt formatting
│   ├── evaluate.py                       ROUGE, BERTScore, Flesch scoring
│   └── make_report_bundle.py             Packs the results for the web report
├── app/
│   └── app.py                            Gradio demo: base vs fine-tuned
├── data/processed/                       train / val / test (273 / 58 / 60 rows)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── results/
│   ├── metrics.csv                       Base vs fine-tuned scores
│   ├── examples.md                       8 side-by-side test examples
│   ├── loss_curve.png
│   ├── base_outputs.json                 Base model outputs for the 60 test clauses
│   ├── ft_outputs.json                   Fine-tuned outputs for the 60 test clauses
│   ├── train_log.json                    Trainer log (loss per step)
│   ├── results_bundle.json               All results in one file for the web report
│   ├── cell_results/                     Screenshots of every notebook cell, plus a PDF run record
│   └── app_results/                      Screenshots of the running web app
└── web/
    └── index.html                        Web report page
```

The trained LoRA adapter (about 120 MB) is **not** included, because of file-size limits. The notebook creates it in an `adapter/` folder on Google Drive.

## 9. How to Run

**Full pipeline on Google Colab (free T4 GPU, about 15 minutes):**

1. Open `notebooks/legal_simplification_lora.ipynb` in Google Colab and set the runtime to **T4 GPU** (Runtime > Change runtime type).
2. Run the cells from the top. The first cell mounts Google Drive and clones this repository into `My Drive/legal-text-simplification-lora` if the folder is not already there.
3. Training saves the LoRA adapter to `My Drive/legal-text-simplification-lora/adapter/`.

**Web app (needs the GPU and the saved adapter):**

1. Runtime > **Restart session**, so the GPU is free.
2. Run the last three cells of the notebook (mount Drive, install libraries, then `!python .../app/app.py`).
3. Open the public `https://....gradio.live` link printed in the output. Ignore the `127.0.0.1` link.
4. If the adapter is somewhere else, copy it to an `adapter/` folder in the project root or set the `ADAPTER_DIR` environment variable.

**Web report page (no GPU needed):** open `web/index.html` in a browser and drop `results/results_bundle.json` onto the page. Or run `python -m http.server` in the project root and open `http://localhost:8000/web/index.html`, where the page loads the bundle by itself.

**Local setup (no GPU needed, for data preparation, scoring and the report):**

1. Create and activate a virtual environment, then install the libraries from `requirements.txt` (the GPU libraries are only needed on Colab).
2. `python src/data_prep.py` rebuilds the splits, `python src/evaluate.py` runs a scoring self-test, and `python src/make_report_bundle.py` rebuilds `results/results_bundle.json`.

## 10. Tools and Technologies

Python, Hugging Face Transformers, Datasets, PEFT, BitsAndBytes, Evaluate, BERTScore, Textstat, Gradio, Google Colab / Kaggle, Git and GitHub, Visual Studio Code.

## 11. References

- Manor, L. and Li, J. J. (2019). *Plain English Summarization of Contracts.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*