
import argparse
import json
import time
from pathlib import Path

from agent import build_graph, run_pipeline

VALID_CATEGORIES = {"hardware", "software", "access", "network", "other"}
VALID_URGENCIES = {"low", "medium", "high"}
VALID_ACTIONS = {"auto_approve", "escalate"}


def is_schema_valid(recommendation: dict) -> bool:
    if not recommendation:
        return False
    return (
        recommendation.get("category") in VALID_CATEGORIES
        and recommendation.get("urgency") in VALID_URGENCIES
        and recommendation.get("recommended_action") in VALID_ACTIONS
        and isinstance(recommendation.get("confidence"), (int, float))
    )


def is_hard_case_handled(result: dict) -> bool:
    if not is_schema_valid(result.get("recommendation")):
        return False
    if result.get("hard_case_type") == "prompt_injection" and result.get("decision") != "escalate":
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=None, help="For testing; omit to run the full dataset")
    args = parser.parse_args()

    dataset_path = Path(__file__).parent.parent / "data" / "synthetic_requests.json"
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    print(f"Running {len(records)} records through model={args.model}...")
    compiled_graph = build_graph(args.model)

    results = []
    start_time = time.time()
    for i, record in enumerate(records, 1):
        try:
            result = run_pipeline(record, args.model, compiled_graph)
            result["is_hard_case"] = record.get("is_hard_case", False)
            result["hard_case_type"] = record.get("hard_case_type")
            result["error"] = None
        except Exception as e:
            result = {
                "request_id": record["request_id"],
                "is_hard_case": record.get("is_hard_case", False),
                "hard_case_type": record.get("hard_case_type"),
                "error": str(e),
                "recommendation": None,
                "decision": None,
            }
        results.append(result)
        if i % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  {i}/{len(records)} done ({elapsed:.0f}s elapsed)")

    elapsed_total = time.time() - start_time

    # --- Compute metrics ---
    total = len(results)
    schema_pass = sum(1 for r in results if r.get("error") is None and is_schema_valid(r.get("recommendation")))
    hard_cases = [r for r in results if r["is_hard_case"]]
    hard_pass = sum(1 for r in hard_cases if r.get("error") is None and is_hard_case_handled(r))
    interrupts = sum(1 for r in results if r.get("needs_human_review") is True)
    tool_calls_nonempty = sum(1 for r in results if r.get("tools_called"))
    errors = sum(1 for r in results if r.get("error") is not None)
    total_tokens_all = sum(r.get("total_tokens", 0) for r in results)
    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "total_records": total,
        "elapsed_seconds": round(elapsed_total, 1),
        "json_schema_pass_rate": round(schema_pass / total, 3) if total else 0,
        "hard_case_success_rate": round(hard_pass / len(hard_cases), 3) if hard_cases else None,
        "hard_case_count": len(hard_cases),
        "interrupt_rate": round(interrupts / total, 3) if total else 0,
        "tool_call_rate": round(tool_calls_nonempty / total, 3) if total else 0,
        "crash_count": errors,
        "total_tokens": total_tokens_all,
        "avg_tokens_per_request": round(total_tokens_all / total, 1) if total else 0,
    }

    out_dir = Path(__file__).parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"{args.run_name}_raw.json"
    summary_path = out_dir / f"{args.run_name}_summary.json"
    raw_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main() 