"""
inference.py – Mandatory judge entry-point for BIS RAG evaluation.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BIS Standards RAG – Inference Script")
    parser.add_argument("--input", required=True, help="Path to input JSON file with queries")
    parser.add_argument("--output", required=True, help="Path to write output JSON results")
    return parser.parse_args()


def load_pipeline():
    """Initialise the RAG pipeline once and reuse."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from rag_pipeline import BISRAGPipeline
        return BISRAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            top_k_retrieve=10,
            top_k_return=5,
            use_llm=True,   # set False if no ANTHROPIC_API_KEY
        )
    except ImportError as e:
        logger.error(f"Failed to import pipeline: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    # ── Load input ──────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries from {input_path}")

    # ── Initialise pipeline ─────────────────────────────────────────────────────
    logger.info("Initialising RAG pipeline …")
    pipeline = load_pipeline()
    logger.info("Pipeline ready.")

    # ── Run inference ───────────────────────────────────────────────────────────
    results = []
    for item in queries:
        qid = item.get("id", item.get("query_id", "unknown"))
        query_text = item.get("query", item.get("description", ""))

        if not query_text:
            logger.warning(f"Empty query for id={qid}, skipping.")
            results.append({
                "id": qid,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            })
            continue

        logger.info(f"Processing query id={qid}: {query_text[:80]}")
        t0 = time.time()
        try:
            response = pipeline.query(query_text)
            latency = round(time.time() - t0, 3)
            results.append({
                "id": qid,
                "retrieved_standards": response["retrieved_standards"],
                "latency_seconds": response.get("latency_seconds", latency),
            })
        except Exception as e:
            logger.error(f"Error on query id={qid}: {e}")
            results.append({
                "id": qid,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            })

    # ── Write output ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results written to {output_path}")
    logger.info(f"Total queries processed: {len(results)}")


if __name__ == "__main__":
    main()
