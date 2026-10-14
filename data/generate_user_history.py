import json
import random
from pathlib import Path
from generate_dataset import HARD_CASES

random.seed(7) # must come after the import: generate_dataset calls seed(42)
                # at module level, and this needs to override it

BASE_SENTENCES_PATH = Path(__file__).parent / "base_sentences.json"
OUT_PATH = Path(__file__).parent / "mock_user_history.json"
N_USERS = random.randint(15, 25)


def load_combined_pool():
    base_sentences = json.loads(BASE_SENTENCES_PATH.read_text(encoding="utf-8"))
    pool = [
        {"issue_text": s["issue_text"], "category": s["intended_category"], "urgency": s["intended_urgency"], "source": "llm_generated"}
        for s in base_sentences
    ]
    pool += [
        {"issue_text": hc["issue_text"], "category": hc["intended_category"], "urgency": hc["intended_urgency"], "source": "hand_written"}
        for hc in HARD_CASES
    ]
    return pool


def main():
    pool = load_combined_pool()
    history = {}
    for i in range(1, N_USERS + 1):
        user_id = f"U-{i:03d}"
        count = random.randint(0, 6)
        history[user_id] = random.sample(pool, min(count, len(pool)))
    OUT_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote mock history for {len(history)} users to {OUT_PATH}")


if __name__ == "__main__":
    main()