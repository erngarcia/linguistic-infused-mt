# scripts/build_parallel_jsonl.py
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build JSONL dataset from parallel .en / .es files.")
    parser.add_argument("--src", required=True, help="Path to source file (.en)")
    parser.add_argument("--tgt", required=True, help="Path to target file (.es)")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    src_path = Path(args.src)
    tgt_path = Path(args.tgt)
    out_path = Path(args.output)

    # read lines
    with src_path.open("r", encoding="utf-8") as f:
        src_lines = [ln.strip() for ln in f if ln.strip()]
    with tgt_path.open("r", encoding="utf-8") as f:
        tgt_lines = [ln.strip() for ln in f if ln.strip()]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Mismatch: {len(src_lines)} source vs {len(tgt_lines)} target lines")

    # write to JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for s, t in zip(src_lines, tgt_lines):
            record = {"src": s, "tgt": t}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(src_lines)} sentence pairs to {out_path}")

if __name__ == "__main__":
    main()
