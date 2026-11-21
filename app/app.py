"""
app.py  -  Stage 8: Gradio demo.

Shows the SAME pasted clause rewritten by (1) the base model and (2) the
fine-tuned model, side by side. Both come from one loaded model: the LoRA
adapter is simply switched off to get the base output.

Needs a GPU, so run it in Colab (after training has saved the adapter):
    !python app/app.py
The adapter folder is read from ../adapter (or the ADAPTER_DIR environment variable).
"""

import os
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from data_prep import build_prompt          # the SAME prompt used in training

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", os.path.join(HERE, "..", "adapter"))

# ---------------- load the model once ----------------
print("Loading model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
tokenizer.padding_side = "left"
print("Ready.")


def generate(clause, use_adapter):
    # The training data was lowercased, so we lowercase the input to match it.
    prompt = build_prompt(clause.lower())
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, return_tensors="pt").to(model.device)

    def run():
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=100, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if use_adapter:
        return run()
    with model.disable_adapter():          # adapter off = original base model
        return run()


def simplify(clause):
    if not clause or not clause.strip():
        return "Please paste a legal clause.", ""
    return generate(clause, use_adapter=False), generate(clause, use_adapter=True)


demo = gr.Interface(
    fn=simplify,
    inputs=gr.Textbox(lines=8, label="Legal clause (paste here)"),
    outputs=[gr.Textbox(label="Base model (no fine-tuning)"),
             gr.Textbox(label="Fine-tuned model (LoRA)")],
    title="Legal Text Simplifier",
    description="Paste a clause from a contract or terms of service. "
                "The output is a plain-English summary, NOT legal advice.",
    examples=[
        ["You may not copy, modify, distribute, sell, or lease any part of our services or included software, "
         "nor may you reverse engineer or attempt to extract the source code of that software."],
        ["We may suspend or terminate your access to the service at any time, without notice, "
         "if we believe you have violated these terms."],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)     # share=True gives a public link when run in Colab
