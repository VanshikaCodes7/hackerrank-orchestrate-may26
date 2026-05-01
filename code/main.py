"""
main.py
Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python main.py                          # Run on support_issues.csv
    python main.py --sample                 # Run on sample_support_issues.csv (for testing)
    python main.py --ask                    # Interactive mode - type your own tickets
    python main.py --input path/to/file.csv # Run on a custom input CSV
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV
from corpus_loader import initialize_corpus
from agent import triage

OUTPUT_FIELDS = ["status", "product_area", "response", "justification", "request_type"]


def run(input_path: Path, output_path: Path):
    # ── 1. Build corpus indexes ──────────────────────────────────────────
    collection, vectorizer, matrix, chunks = initialize_corpus()

    # ── 2. Read input CSV ────────────────────────────────────────────────
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_fields = reader.fieldnames or []

    print(f"[Main] Processing {len(rows)} tickets from {input_path.name}...\n")

    # ── 3. Process each row ──────────────────────────────────────────────
    results = []
    errors = 0

    for i, row in enumerate(tqdm(rows, desc="Triaging tickets", unit="ticket")):
        # Support both capitalised (Issue, Subject, Company) and lowercase column names
        issue   = row.get("Issue") or row.get("issue") or ""
        subject = row.get("Subject") or row.get("subject") or ""
        company = row.get("Company") or row.get("company") or "None"

        try:
            result = triage(
                issue=issue,
                subject=subject,
                company=company,
                collection=collection,
                vectorizer=vectorizer,
                matrix=matrix,
                chunks=chunks,
            )
        except Exception as e:
            print(f"\n  [ERROR] Row {i+1} failed: {e}")
            result = {
                "status": "escalated",
                "product_area": "Unknown",
                "response": "A human support agent will review this ticket.",
                "justification": f"Processing error: {str(e)[:100]}",
                "request_type": "product_issue",
            }
            errors += 1

        merged = {**row, **result}
        results.append(merged)

        time.sleep(0.3)

    # ── 4. Write output CSV ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_fields = list(input_fields) + [f for f in OUTPUT_FIELDS if f not in input_fields]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ── 5. Summary ───────────────────────────────────────────────────────
    replied   = sum(1 for r in results if r.get("status") == "replied")
    escalated = sum(1 for r in results if r.get("status") == "escalated")

    print(f"\n{'='*50}")
    print(f"✓ Done! Results written to: {output_path}")
    print(f"  Total tickets : {len(rows)}")
    print(f"  Replied       : {replied}")
    print(f"  Escalated     : {escalated}")
    if errors:
        print(f"  Errors        : {errors}")
    print(f"{'='*50}\n")


def ask_interactive():
    """Interactive mode — type support tickets directly in the terminal."""
    print("\n[Corpus Loader] Loading corpus...")
    collection, vectorizer, matrix, chunks = initialize_corpus()

    print("=" * 50)
    print("  Support Triage — Interactive Mode")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        print()
        company = input("Company (hackerrank / claude / visa) [Enter to skip]: ").strip() or "None"
        subject = input("Subject: ").strip()
        issue   = input("Issue: ").strip()

        if issue.lower() in ("exit", "quit") or subject.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not issue:
            print("[WARN] Issue cannot be empty, try again.")
            continue

        print("\nTriaging...\n")
        result = triage(
            issue=issue,
            subject=subject,
            company=company,
            collection=collection,
            vectorizer=vectorizer,
            matrix=matrix,
            chunks=chunks,
        )

        print(f"  Status       : {result['status'].upper()}")
        print(f"  Product Area : {result['product_area']}")
        print(f"  Request Type : {result['request_type']}")
        print(f"\n  Response:\n  {result['response']}")
        print(f"\n  Justification:\n  {result['justification']}")
        print("\n" + "-" * 50)


def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Run on sample CSV (for testing)")
    parser.add_argument("--ask", action="store_true", help="Interactive mode — type your own tickets")
    parser.add_argument("--input", type=str, help="Custom input CSV path")
    parser.add_argument("--output", type=str, help="Custom output CSV path")
    args = parser.parse_args()

    if args.ask:
        ask_interactive()
        return

    if args.input:
        input_path = Path(args.input)
    elif args.sample:
        input_path = SAMPLE_CSV
    else:
        input_path = INPUT_CSV

    output_path = Path(args.output) if args.output else OUTPUT_CSV

    run(input_path, output_path)


if __name__ == "__main__":
    main()
