#  MT-GenderBias Experiments (T0–T2)

This repository contains scripts and datasets for reproducing experiments on **Linguistic Knowledge-Infused Fine-Tuning for Mitigating Gender Bias in Machine Translation** using Mistral and Salamandrata models.  
The project extends MT-GenEval with linguistically informed data (T2) to evaluate how explicit reasoning about gender and syntax affects translation behavior.

---

##  Project structure

```
src/
├── data/
│   ├── augment_mtgeneval.py       # augment MT-GenEval .dev files using OpenAI LLMs
│   ├── templates.py               # inclues a template to augment MT-GenEval
|   ├── build_parallel_set.py      #builds T1 dataset. Parallel set src-trg         
├── training/
│   └── train_models.py        # unified training script for T1/T2 across models
└── inference/
    └── run_inference.py           # unified inference script for T0/T1/T2
```

---

## Experimental setup

| Task | Description | Model | Notes |
|------|--------------|--------|-------|
| **T0** | Zero-shot translation baseline | Mistral / Salamandrata | No adapter — base model |
| **T1** | Fine-tuned on standard MT data | Mistral / Salamandrata | Tuned for translation |
| **T2** | Fine-tuned on reasoning-augmented data | Mistral / Salamandrata | Uses linguistic prompts (pronoun resolution, actors, gender reasoning) |

---

##  1. Data preparation

### Step 1 — Augment MT-GenEval `.dev` files
Generate reasoning-augmented examples (T2 data) using OpenAI’s API:

```bash
python src/data/augment_mtgeneval.py     --input_dir data/mtgeneval/en_es/dev/     --output_path data/augmented/t2_train_raw.jsonl     --model gpt-4o-mini
```

Each entry is stored as:

```json
{
  "source": "Maria called her brother and said she would arrive late.",
  "reasoning": "0. Count sentences: 1 ...",
  "translation": "María llamó a su hermano y dijo que ella llegaría tarde."
}
```

---

### Step 2 — Prepare training-ready JSONL (T2)

```bash
python src/data/prep_t2_dataset.py     --raw_jsonl data/augmented/t2_train_raw.jsonl     --out_dir data/prepared     --on_missing_translation wrap_last_para     --dedupe_by_source
```

This script:
- validates `<source_text>`, `<reasoning>`, `<translation>` blocks  
- applies normalization and deduplication  
- outputs `t2_train.jsonl` and `t2_dev.jsonl`

---

## 2. Training

A single script trains all (T1/T2 × model) combinations:

```bash
python src/training/train_all_models.py     --model mistral     --task t2     --lang es     --upload_s3
```

### Available arguments
| Flag | Description |
|------|--------------|
| `--model` | `mistral` or `salamandrata` |
| `--task` | `t1` (translation) or `t2` (reasoning) |
| `--lang` | target language (`es` or `ca`) |
| `--upload_s3` | upload results to your S3 bucket (optional) |

The script:
1. Loads LoRA adapters on top of quantized base models (4-bit NF4)
2. Trains using `trl.SFTTrainer`
3. Logs metrics to **Weights & Biases**
4. Optionally uploads checkpoints to S3

---

## 3. Inference

Run zero-shot (T0), fine-tuned (T1), or reasoning-augmented (T2) translation using one command.

### Single sentence
```bash
python src/inference/run_inference.py   --model mistral   --task t2   --lang es   --adapter_dir ./runs/mistral_t2_adapter   --text "Maria called her brother and said she would arrive late."
```

### Batch from file
```bash
python src/inference/run_inference.py   --model salamandrata   --task t1   --lang es   --adapter_dir ./runs/salamandrata_t1_adapter   --input_file data/eval/test.en   --output_file runs/infer/test.t1.salamandrata.es
```

### Zero-shot baseline (T0)
```bash
python src/inference/run_inference.py   --model mistral   --task t0   --lang es   --text "The nurse said he was tired."
```

No adapter is required for `--task t0`.

---

## Prompt styles by task

| Task | Prompt behavior |
|------|------------------|
| **T0** | Simple instruction: *“Translate the following text into Spanish.”* |
| **T1** | Explicit instruction: *“Translate the following. Keep named entities intact.”* |
| **T2** | Linguistic reasoning prompt: *“Reason about pronoun resolution and actors before translating.”* |

---

## 4. Optional — Upload/download from S3

Each training and inference script can interact with S3:

```bash
--s3_bucket <your_bucket> --s3_prefix t2_mistral_spanish
```

During inference, this automatically downloads the LoRA adapter before loading it.

---

## Outputs

- Fine-tuned adapters:  
  `runs/{model}_{task}_{lang}/checkpoint-*`
- Inference results:  
  `runs/infer/{context}/{model}/t{task}_{lang}.es`
- Logs:  
  Weights & Biases dashboard

---

## Citation / Paper context

This pipeline reproduces the reasoning-augmented setup described in Section §4.4.1 of the *Linguistic Knowledge-Infused Fine-Tuning for Mitigating Gender Bias in Machine Translation* paper.  
The goal is to evaluate whether adding structured linguistic reasoning (pronoun resolution, actor identification, subordination analysis) improves gender-balanced translation.

---

## Setup

Install dependencies:

```bash
conda env create -f environment.yml
```

Recommended environment:

- Python ≥ 3.10  
- CUDA-compatible GPU (A100, V100, or RTX 3090/4090)
- Access to OpenAI API (for augmentation step)
- Weights & Biases account (for logging)


---

## License

MIT License.  
© 2025 Ernesto Garcia Estrada. All rights reserved.
