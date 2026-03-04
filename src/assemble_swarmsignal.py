#!/usr/bin/env python3
"""SwarmSignal Training Data Assembler
=======================================

Assembles cooked AI trends writer pairs into training-ready JSONL.

Input:  /tmp/swarmsignal_cook/cat_*.jsonl (from cook_swarmsignal.py)
Output: swarmsignal_train.jsonl + swarmsignal_eval.jsonl

Usage:
    python3 -m src.assemble_swarmsignal
    python3 -m src.assemble_swarmsignal --status
    python3 -m src.assemble_swarmsignal --sample 5
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from glob import glob
from pathlib import Path

INPUT_DIR = Path("/tmp/swarmsignal_cook")
OUTPUT_DIR = INPUT_DIR  # train/eval written alongside cook output

CATEGORIES = ["briefing", "trends", "company", "culture", "market", "journal"]


def load_pairs() -> list:
    """Load all cooked pairs, dedup by fingerprint."""
    pairs = []
    seen = set()

    for category in CATEGORIES:
        cat_file = INPUT_DIR / f"cat_{category}.jsonl"
        if not cat_file.exists():
            continue
        with open(cat_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)

                # Dedup by fingerprint
                fp = rec.get("metadata", {}).get("fingerprint", "")
                if not fp:
                    # Generate fingerprint from content
                    content = rec.get("messages", [{}])[-1].get("content", "")
                    fp = hashlib.sha256(content[:200].encode()).hexdigest()[:16]
                if fp in seen:
                    continue
                seen.add(fp)

                # Validate: must have 3 messages (system, user, assistant)
                msgs = rec.get("messages", [])
                if len(msgs) < 3:
                    continue

                # Content quality — assistant response must be substantive
                assistant = msgs[-1].get("content", "")
                if len(assistant) < 300:
                    continue

                pairs.append(rec)

    return pairs


def assemble(eval_pct: float = 0.05, seed: int = 42):
    """Assemble training and eval sets."""
    print("Loading cooked SwarmSignal pairs...")
    pairs = load_pairs()
    print(f"  Loaded: {len(pairs):,} unique pairs")

    if not pairs:
        print("  ERROR: No pairs found. Run cook_swarmsignal.py first.")
        return

    # Category distribution
    cat_counts = Counter(
        r.get("metadata", {}).get("category", "unknown")
        for r in pairs
    )
    print(f"\n  Category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"    {cat:<15} {count:>6,}")

    # Tier distribution (gen vs rewrite)
    tier_counts = Counter(
        r.get("metadata", {}).get("tier", "unknown")
        for r in pairs
    )
    print(f"\n  Tier distribution:")
    for tier, count in tier_counts.most_common():
        print(f"    {tier:<15} {count:>6,}")

    # Content length stats
    lengths = [len(r["messages"][-1]["content"]) for r in pairs]
    avg_len = sum(lengths) / len(lengths)
    print(f"\n  Avg response length: {avg_len:,.0f} chars")
    print(f"  Min: {min(lengths):,} | Max: {max(lengths):,}")

    # Split train/eval — stratified by category
    random.seed(seed)
    random.shuffle(pairs)

    eval_count = max(500, int(len(pairs) * eval_pct))
    eval_pairs = pairs[:eval_count]
    train_pairs = pairs[eval_count:]

    # Verify eval has category diversity
    eval_cats = Counter(r.get("metadata", {}).get("category", "") for r in eval_pairs)
    print(f"\n  Train: {len(train_pairs):,} pairs")
    print(f"  Eval:  {len(eval_pairs):,} pairs")
    print(f"  Eval categories: {dict(eval_cats)}")

    # Write
    train_file = OUTPUT_DIR / "swarmsignal_train.jsonl"
    eval_file = OUTPUT_DIR / "swarmsignal_eval.jsonl"

    with open(train_file, "w") as f:
        for rec in train_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(eval_file, "w") as f:
        for rec in eval_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_size = train_file.stat().st_size / (1024 * 1024)
    eval_size = eval_file.stat().st_size / (1024 * 1024)

    print(f"\n  Output:")
    print(f"    Train: {train_file} ({train_size:.1f} MB)")
    print(f"    Eval:  {eval_file} ({eval_size:.1f} MB)")
    print(f"\n  Ready for SCP to swarmrails:/data2/swarmsignal-9b/data/")


def status():
    """Show current cook output status."""
    pairs = load_pairs()
    cat_counts = Counter(
        r.get("metadata", {}).get("category", "unknown")
        for r in pairs
    )
    print(f"SwarmSignal pairs available: {len(pairs):,}")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:<15} {count:>6,}")

    # Check cook progress
    progress_file = INPUT_DIR / "progress.json"
    if progress_file.exists():
        prog = json.loads(progress_file.read_text())
        print(f"\nCook: {prog['total_written']:,}/{prog['total_target']:,}")
        print(f"Rate: {prog['rate_per_min']}/min | ETA: {prog['eta_hours']}h")


def sample(n: int = 5):
    """Show N random samples."""
    pairs = load_pairs()
    if not pairs:
        print("No pairs found.")
        return

    samples = random.sample(pairs, min(n, len(pairs)))
    for i, rec in enumerate(samples, 1):
        cat = rec.get("metadata", {}).get("category", "?")
        tier = rec.get("metadata", {}).get("tier", "?")
        prompt = rec["messages"][1]["content"][:100]
        response = rec["messages"][-1]["content"][:200]
        print(f"\n{'='*60}")
        print(f"  Sample {i}/{n} — {cat} ({tier})")
        print(f"  Prompt: {prompt}...")
        print(f"  Response: {response}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--eval-pct", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0, help="Show N random samples")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.sample > 0:
        sample(args.sample)
    else:
        assemble(eval_pct=args.eval_pct, seed=args.seed)
