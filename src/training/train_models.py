"""
Unified fine-tuning script for all (T1/T2 × model) combinations.
Reports trainable parameters of LoRA

Examples:
    python src/training/train_all_models.py --model mistral --task t1 --lang es
    python src/training/train_all_models.py --model salamandrata --task t2 --lang es --upload_s3
"""

import os
import boto3
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv


def upload_to_s3(local_dir: str, bucket_name: str, s3_prefix: str):
    """Upload all files in a directory to an S3 bucket/prefix."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            logging.info(f"Uploading {local_path} → s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logging.info(f"🔹 Trainable params: {trainable:,d}/{total:,d} "
                 f"({100 * trainable / total:.2f}%)")




MODEL_REGISTRY = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "default_output": "runs/mistral",
    },
    "salamandrata": {
        "name": "BSC-LT/salamandrata-7b",
        "default_output": "runs/salamandrata",
    },
}

TASK_CONFIG = {
    "t1": {
        "prompt_type": "plain_translation",
        "dataset": "data/prepared/prompted_mistral_train_t1.jsonl",
        "s3_prefix": "t1_{model}_{lang}",
    },
    "t2": {
        "prompt_type": "linguistic_reasoning",
        "dataset": "data/prepared/t2_train_with_xml.jsonl",
        "s3_prefix": "t2_{model}_{lang}",
    },
}


def train_model(model_key: str, task_key: str, lang: str = "es", upload_s3: bool = False):
    model_cfg = MODEL_REGISTRY[model_key]
    task_cfg = TASK_CONFIG[task_key]

    model_name = model_cfg["name"]
    dataset_path = task_cfg["dataset"]
    output_dir = Path(model_cfg["default_output"]) / f"{task_key}_{lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Training {model_key.upper()} on {task_key.upper()} ({lang})")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Output:  {output_dir}")

    dataset = load_dataset("json", data_files=dataset_path)
    logging.info(f"Loaded {len(dataset['train'])} samples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_cfg
    )
    print_trainable_parameters(model)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to=["wandb"],
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_cfg,
        args=sft_cfg,
    )

    logging.info("Starting training")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Saved fine-tuned model to {output_dir}")


    if upload_s3:
        load_dotenv()
        bucket = os.getenv("BUCKET_NAME")
        prefix = task_cfg["s3_prefix"].format(model=model_key, lang=lang)
        upload_to_s3(str(output_dir), bucket, prefix)
        logging.info(f"Uploaded to s3://{bucket}/{prefix}")



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MT gender-bias models.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--task", choices=list(TASK_CONFIG.keys()), required=True)
    parser.add_argument("--lang", default="es", help="Language code (es or ca)")
    parser.add_argument("--upload_s3", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    train_model(args.model, args.task, args.lang, args.upload_s3)

if __name__ == "__main__":
    main()
