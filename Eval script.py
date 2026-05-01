"""
eval_script.py – Official evaluation script for BIS Standards RAG.

Computes:
  • Hit Rate @3  – % queries where ≥1 expected standard appears in top-3
  • MRR @5       – Mean Reciprocal Rank of first correct standard in top-5
  • Avg Latency  – Average response time per query

Usage:
    python eval_script.py --predictions results.json --ground_truth ground_truth.json
    python eval_script.py --predictions results.json --ground_truth ground_truth.json --output metrics.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="BIS RAG Evaluation Script")
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions JSON (list of {id, retrieved_standards, latency_seconds})")
    parser.add_argument("--ground_truth", required=True,
                        help="Path to ground truth JSON (list of {id, expected_standards})")
    parser.add_argument("--output", default=None,
                        help="Optional path to write metrics JSON")
    return parser.parse_args()


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hit_at_k(retrieved: list, expected: list, k: int = 3) -> int:
    """Return 1 if any expected standard appears in the top-k retrieved."""
    top_k = [r.strip().upper() for r in retrieved[:k]]
    expected_set = {e.strip().upper() for e in expected}
    return int(bool(top_k and expected_set.intersection(top_k)))


def reciprocal_rank(retrieved: list, expected: list, k: int = 5) -> float:
    """Return 1/rank of first expected standard in top-k, or 0."""
    top_k = [r.strip().upper() for r in retrieved[:k]]
    expected_set = {e.strip().upper() for e in expected}
    for rank, std_id in enumerate(top_k, start=1):
        if std_id in expected_set:
            return 1.0 / rank
    return 0.0


def evaluate(predictions: list, ground_truth: list) -> dict:
    # Index ground truth by id
    gt_index = {}
    for item in ground_truth:
        qid = str(item.get("id", item.get("query_id", "")))
        expected = item.get("expected_standards", item.get("relevant_standards", []))
        gt_index[qid] = expected

    hits_3 = []
    rr_5 = []
    latencies = []
    missing = []

    for pred in predictions:
        qid = str(pred.get("id", ""))
        retrieved = pred.get("retrieved_standards", [])
        latency = pred.get("latency_seconds", 0.0)

        if qid not in gt_index:
            missing.append(qid)
            continue

        expected = gt_index[qid]
        hits_3.append(hit_at_k(retrieved, expected, k=3))
        rr_5.append(reciprocal_rank(retrieved, expected, k=5))
        latencies.append(float(latency))

    n = len(hits_3)
    if n == 0:
        print("ERROR: No matching query IDs found between predictions and ground truth.")
        sys.exit(1)

    hit_rate = (sum(hits_3) / n) * 100
    mrr = sum(rr_5) / n
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    metrics = {
        "num_queries_evaluated": n,
        "hit_rate_at_3": round(hit_rate, 2),          # percentage
        "mrr_at_5": round(mrr, 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "missing_ids": missing,
        "targets": {
            "hit_rate_at_3_target": ">80%",
            "mrr_at_5_target": ">0.7",
            "avg_latency_target": "<5 seconds",
        },
        "passed": {
            "hit_rate_at_3": hit_rate > 80,
            "mrr_at_5": mrr > 0.7,
            "avg_latency": avg_latency < 5.0,
        },
    }
    return metrics


def print_report(metrics: dict):
    print("\n" + "=" * 55)
    print("  BIS RAG Evaluation Report")
    print("=" * 55)
    print(f"  Queries evaluated : {metrics['num_queries_evaluated']}")
    print("-" * 55)

    def status(passed): return "✅ PASS" if passed else "❌ FAIL"

    print(f"  Hit Rate @3       : {metrics['hit_rate_at_3']:>6.2f}%   "
          f"(target >80%)  {status(metrics['passed']['hit_rate_at_3'])}")
    print(f"  MRR @5            : {metrics['mrr_at_5']:>6.4f}    "
          f"(target >0.7)  {status(metrics['passed']['mrr_at_5'])}")
    print(f"  Avg Latency       : {metrics['avg_latency_seconds']:>6.3f}s   "
          f"(target <5s)   {status(metrics['passed']['avg_latency'])}")

    if metrics["missing_ids"]:
        print(f"\n  ⚠ Missing IDs: {metrics['missing_ids']}")

    all_pass = all(metrics["passed"].values())
    print("=" * 55)
    print(f"  Overall: {'✅ ALL TARGETS MET' if all_pass else '⚠ SOME TARGETS MISSED'}")
    print("=" * 55 + "\n")


def main():
    args = parse_args()

    predictions = load_json(args.predictions)
    ground_truth = load_json(args.ground_truth)

    metrics = evaluate(predictions, ground_truth)
    print_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to: {args.output}")

    # Exit with non-zero if any target missed
    if not all(metrics["passed"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
