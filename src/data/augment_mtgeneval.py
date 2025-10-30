# src/data/augment_mtgeneval.py
import os
import json
import time
import openai
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.data.templates import PROMPT_TEMPLATE
import os
from dotenv import load_dotenv
# Load variables from .env file

def augment_mtgeneval(
    input_dir: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    lang_name: str = "Spanish",
    rate_limit: float = 1.5,
):
    """
    Build the linguistically infused dataset by calling OpenAI o440 model with reasoning prompts.
    """

    input_dir = Path(input_dir)
    src_file = next(input_dir.glob("*.en"))
    tgt_file = next(input_dir.glob("*.es"))
    df_src = pd.read_csv(src_file, names=["source"], sep="\n", quoting=3)
    df_tgt = pd.read_csv(tgt_file, names=["target"], sep="\n", quoting=3)
    df = pd.concat([df_src, df_tgt], axis=1)

    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = PROMPT_TEMPLATE.format(sentence=row["source"], lang_name=lang_name)

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000, #value determined by trail and error to avoid truncated output
            )
            completion = response.choices[0].message.content
            # Try to extract JSON from the model output
            start = completion.find("{")
            end = completion.rfind("}") + 1
            parsed = json.loads(completion[start:end])

            results.append({
                "source": row["source"],
                "target_ref": row["target"],
                "reasoning": parsed.get("reasoning", ""),
                "translation": parsed.get("translation", "")
            })

        except Exception as e:
            print(f"Error on example: {row['source'][:60]} | {e}")
            continue

        time.sleep(rate_limit)  # respect rate limits

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} augmented samples to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--lang_name", default="Spanish")
    parser.add_argument("--rate_limit", type=float, default=1.5)
    args = parser.parse_args()

    augment_mtgeneval(
        args.input_dir,
        args.output_path,
        args.model,
        args.lang_name,
        args.rate_limit,
    )
