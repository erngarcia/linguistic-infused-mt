"""
Unified inference script for MT models (T0 = zero-shot, T1 = translation, T2 = reasoning-augmented).

Examples:
    # Zero-shot baseline (T0)
    python src/inference/run_inference.py \
        --model mistral --task t0 --lang es \
        --text "Maria called her brother and said she would arrive late."

    # Fine-tuned T1
    python src/inference/run_inference.py \
        --model mistral --task t1 --lang es \
        --adapter_dir ./runs/mistral_t1_adapter \
        --input_file data/test/test.en \
        --output_file runs/infer/test.t1.mistral.es

    # Fine-tuned T2 with reasoning
    python src/inference/run_inference.py \
        --model salamandrata --task t2 --lang es \
        --adapter_dir ./runs/salamandrata_t2_adapter \
        --input_file data/test/test.en \
        --output_file runs/infer/test.t2.salamandrata.es
"""

import os
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Optional helper for downloading from S3
try:
    from src.utils.s3_utils import download_from_s3
except ImportError:
    download_from_s3 = None

# Available models
MODEL_REGISTRY = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "salamandrata": "BSC-LT/salamandrata-7b",
}

# Templates
PROMPTS = {
    "t0": lambda text, lang_name: f"""Translate the following text into {lang_name}.
Make sure the translation is accurate and natural.

Source:
{text}

Translation:
""",

    "t1": lambda text, lang_name: f"""Translate the following into {lang_name}.
Rules:
- Only output the translation (no explanations).
- Keep named entities intact.
- Preserve meaning and tone.

Source:
{text}

Translation:
""",

    "t2": lambda text, lang_name: f"""You are a linguist translating into {lang_name}.
Before translating, reason through pronoun resolution and actor identification.
Only output the final translation (no reasoning).

Source:
{text}

Translation:
""",
}

def load_base_model(base_model_name: str):
    """Load base model with 4-bit quantization (for T0)."""
    logging.info(f"Loading base model: {base_model_name}")
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    return model


def load_adapter_model(base_model_name: str, adapter_path: str):
    """Load base model and merge LoRA adapter (for T1/T2)."""
    base = load_base_model(base_model_name)
    logging.info(f"Merging adapter from {adapter_path}")
    return PeftModel.from_pretrained(base, adapter_path)

def generate_translation(
    text: str,
    model,
    tokenizer,
    prompt_template,
    lang_name="Spanish",
    max_new_tokens=300,
    temperature=0.1,
):
    """Generate a translation for one sentence."""
    prompt = prompt_template(text, lang_name)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def batch_infer(
    model,
    tokenizer,
    lines: List[str],
    prompt_template,
    lang_name="Spanish",
    max_new_tokens=300,
):
    """Run inference over multiple sentences."""
    translations = []
    for line in tqdm(lines, desc="Translating"):
        if not line.strip():
            translations.append("")
            continue
        translation = generate_translation(
            line, model, tokenizer, prompt_template, lang_name, max_new_tokens
        )
        translations.append(translation)
    return translations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with fine-tuned or zero-shot models.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--task", choices=["t0", "t1", "t2"], required=True)
    parser.add_argument("--lang", choices=["es", "ca"], default="es")
    parser.add_argument("--adapter_dir", help="Path to local adapter directory (required for t1/t2).")
    parser.add_argument("--text", help="Single sentence to translate.")
    parser.add_argument("--input_file", help="File with one sentence per line.")
    parser.add_argument("--output_file", help="Optional output file for batch mode.")
    parser.add_argument("--s3_bucket", help="Optional S3 bucket name.")
    parser.add_argument("--s3_prefix", help="Optional S3 prefix for download.")
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    lang_name = "Spanish" if args.lang == "es" else "Catalan"

    base_model_name = MODEL_REGISTRY[args.model]

    # Handle S3 download if specified
    if args.s3_bucket and args.s3_prefix and download_from_s3:
        if not args.adapter_dir:
            raise SystemExit("adapter_dir must be specified when downloading from S3.")
        download_from_s3(args.s3_bucket, args.s3_prefix, args.adapter_dir)

    # Load model depending on task
    if args.task == "t0":
        model = load_base_model(base_model_name)
    else:
        if not args.adapter_dir or not os.path.exists(args.adapter_dir):
            raise SystemExit("adapter_dir is required for t1/t2 tasks.")
        model = load_adapter_model(base_model_name, args.adapter_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_template = PROMPTS[args.task]

    if args.text:
        translation = generate_translation(
            args.text, model, tokenizer, prompt_template, lang_name
        )
        print(f"\nTranslation:\n{translation}")
        return

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        translations = batch_infer(
            model, tokenizer, lines, prompt_template, lang_name
        )

        if args.output_file:
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                for t in translations:
                    f.write(t + "\n")
            logging.info(f"Saved translations to {args.output_file}")
        else:
            print("\n".join(translations))
    else:
        logging.error("Please provide either --text or --input_file.")


if __name__ == "__main__":
    main()
