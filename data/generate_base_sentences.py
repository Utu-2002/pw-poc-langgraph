
import argparse
import json
import re
from pathlib import Path

import ollama

CATEGORIES = ["hardware", "software", "access", "network", "other"]
URGENCIES = ["low", "medium", "high"]
SENTENCES_PER_COMBO = 4  

def generate_for_combo(model: str, category: str, urgency: str, n: int) -> list[str]:
    prompt = f"""Write {n} short, realistic IT/service requests a company employee
might send, all in category "{category}" with "{urgency}" urgency. One per line,
no numbering, no extra text, just the request sentences themselves."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    lines = response["message"]["content"].strip().split("\n")
    # clean up: drop empty lines and any leading numbering like "1." or "-"
    cleaned = [re.sub(r"^[\d\.\-\)\s]+", "", line).strip() for line in lines]
    return [line for line in cleaned if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    pool = []
    for category in CATEGORIES:
        for urgency in URGENCIES:
            sentences = generate_for_combo(args.model, category, urgency, SENTENCES_PER_COMBO)
            for text in sentences:
                pool.append({
                    "issue_text": text, "intended_category": category, "intended_urgency": urgency,
                })
            print(f"{category}/{urgency}: {len(sentences)} sentences generated")

    out_path = Path(__file__).parent / "base_sentences.json"
    out_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(pool)} base sentences to {out_path}")


if __name__ == "__main__":
    main()