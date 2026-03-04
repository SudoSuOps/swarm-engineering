#!/usr/bin/env python3
"""SwarmSignal Cascade Cook — 6K Intelligence Cascade Pairs
=============================================================

Targeted cook for the 3 intelligence cascade formats only:
  - SIGNAL_EXTRACTION (2,500 pairs)
  - GAP_DETECTION (2,000 pairs)
  - TREND_SYNTHESIS (1,500 pairs)

Uses the same Maverick two-tier architecture as v2 cook.
Output goes to /tmp/swarmsignal_cascade/ for Phase 2 assembly.

Usage:
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_cascade
"""

import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
GEN_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
PASS_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
WORKERS = 50
SEED = 4026
OUT_DIR = Path("/tmp/swarmsignal_cascade")
MIN_QUALITY_LEN = 300

TARGETS = {
    "signal_extraction": 9000,
    "gap_detection": 7500,
    "trend_synthesis": 7500,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT SPECS
# ═══════════════════════════════════════════════════════════════════════════════

FORMATS = {
    "SIGNAL_EXTRACTION": {
        "instruction": (
            "FORMAT: SIGNAL_EXTRACTION\n"
            "Analyze the following AI/data infrastructure development and extract a structured signal.\n"
            "Your output MUST use these exact headers:\n\n"
            "SIGNAL: [One-sentence description of the weak signal detected]\n"
            "WHY IT MATTERS: [2-3 sentences on implications for AI infrastructure]\n"
            "DATA OPPORTUNITY: [Specific dataset or training data that should be created]\n"
            "CURATION SCORE: [1-10 rating of how actionable this signal is for dataset curation]\n"
        ),
        "min_len": 300,
        "max_tokens": 1536,
    },
    "GAP_DETECTION": {
        "instruction": (
            "FORMAT: GAP_DETECTION\n"
            "Analyze current AI model capabilities in the described domain and identify a specific gap.\n"
            "Your output MUST use these exact headers:\n\n"
            "CURRENT CAPABILITIES: [What models can do today in this area]\n"
            "KNOWN WEAKNESSES: [Specific failure modes or blind spots]\n"
            "DATA GAP: [What training data is missing that causes this weakness]\n"
            "CURATION OBJECT: [Precise description of the dataset needed to fill this gap, "
            "including size estimate and format]\n"
        ),
        "min_len": 500,
        "max_tokens": 2048,
    },
    "TREND_SYNTHESIS": {
        "instruction": (
            "FORMAT: TREND_SYNTHESIS\n"
            "Synthesize multiple signals into a Top-5 list of the most important dataset "
            "curation opportunities right now.\n"
            "For each of the 5 items, use these exact headers:\n\n"
            "NAME: [Dataset/curation object name]\n"
            "DESCRIPTION: [What this dataset contains and why it matters]\n"
            "WHY NOW: [Why this is urgent/timely]\n"
            "TARGET PAIRS: [Estimated number of training pairs needed]\n"
            "EXPECTED MODEL IMPACT: [How a model trained on this data would improve]\n"
        ),
        "min_len": 800,
        "max_tokens": 2560,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

SIGNAL_TOPICS = [
    # Infrastructure signals
    "NVIDIA just released a new GPU architecture targeting inference workloads with 2x memory bandwidth",
    "A startup raised $50M to build custom ASIC chips specifically for mixture-of-experts inference",
    "Google DeepMind published a paper showing 10x training efficiency gains from curriculum learning",
    "Meta open-sourced a new 400B parameter model with Apache 2.0 license",
    "A new quantization method achieves 2-bit weights with less than 1% quality loss",
    "AMD's MI400 GPU is showing competitive MLPerf benchmarks against H100",
    "A university lab demonstrated federated learning across 1000 edge devices with no accuracy loss",
    "Cloudflare launched serverless GPU inference at $0.001 per request",
    "A new paper shows transformers can be replaced with state-space models for 5x faster inference",
    "Intel's Gaudi 3 is being adopted by three major cloud providers for inference",
    # Data/curation signals
    "Hugging Face reported 500K new datasets uploaded in the last quarter",
    "A new benchmark shows synthetic data trained models outperform human-curated on 7 of 10 tasks",
    "The EU AI Act now requires training data provenance documentation for all foundation models",
    "A startup built an automated data quality scoring system that reduced curation costs by 80%",
    "Research shows models trained on chain-of-thought data show 40% better reasoning",
    "Wikipedia announced a partnership with AI labs to create structured knowledge extraction datasets",
    "A new tool can automatically detect and remove PII from training datasets at scale",
    "The first on-chain dataset marketplace launched with 10K verified datasets",
    "Medical imaging datasets with expert annotations reached 1M images for the first time",
    "Financial document understanding benchmarks show current models fail on 60% of complex tables",
    # Industry signals
    "Three major banks formed a consortium to create a shared financial AI training dataset",
    "The US DoD awarded a $200M contract for sovereign AI infrastructure on domestic soil",
    "A drone manufacturer is building onboard AI using 4-bit quantized models for real-time navigation",
    "Agricultural AI models trained on satellite imagery achieved 95% crop disease detection",
    "Legal AI startups raised $2B collectively in Q4, focusing on document analysis",
    "Robotics foundation models are being trained on 10M hours of manipulation video data",
    "Autonomous vehicle companies are sharing a 50PB driving dataset through a new consortium",
    "Climate modeling AI now uses 100B parameter models trained on 40 years of satellite data",
    "Pharmaceutical AI reduced drug discovery timelines from 4 years to 8 months in a clinical trial",
    "Edge AI chipmakers shipped 500M units in 2025, up 300% from 2024",
    # Sovereign/blockchain signals
    "A blockchain protocol launched verified compute proofs for AI training runs",
    "Three countries announced national AI data sovereignty requirements for 2026",
    "Decentralized GPU networks now offer 40% cost savings over centralized cloud for training",
    "A new token standard enables fractional ownership of trained AI models",
    "On-chain model registries now track 50K model versions with full training provenance",
    # Culture/builder signals
    "The vibe coding movement shows 10x developer productivity but 5x more bugs in production",
    "Open-source AI labs collectively employ more researchers than any single tech company",
    "A 19-year-old built a 7B parameter model in their garage using consumer GPUs",
    "The maker community is building custom Jetson clusters for local AI inference",
    "AI pair programming tools now write 60% of code at Fortune 500 companies",
]

GAP_DOMAINS = [
    "commercial real estate underwriting and deal analysis",
    "pharmaceutical drug interaction detection and safety monitoring",
    "aviation maintenance prediction and flight operations",
    "medical diagnosis from multimodal patient records",
    "financial document understanding including complex tables and footnotes",
    "legal contract analysis and risk assessment",
    "supply chain optimization under uncertainty",
    "climate risk modeling for infrastructure planning",
    "agricultural pest and disease detection from drone imagery",
    "manufacturing quality control from sensor data",
    "cybersecurity threat detection from network logs",
    "autonomous vehicle edge case handling in adverse weather",
    "code generation for embedded systems and microcontrollers",
    "scientific paper comprehension and hypothesis generation",
    "multi-language document translation preserving domain terminology",
    "time series anomaly detection in industrial IoT",
    "geospatial intelligence from satellite and aerial imagery",
    "energy grid optimization and demand forecasting",
    "insurance claims processing and fraud detection",
    "educational content personalization and assessment",
]

TREND_CONTEXTS = [
    "the AI infrastructure market in Q1 2026 with record GPU shipments and new quantization breakthroughs",
    "the explosion of domain-specific AI models in healthcare, finance, and legal sectors",
    "the shift from cloud-only to edge-cloud hybrid AI architectures",
    "the growing demand for verified, high-quality training datasets with provenance",
    "the emergence of AI agent frameworks that require structured tool-use training data",
    "the sovereign AI movement with countries building domestic compute and data infrastructure",
    "the convergence of blockchain and AI for model provenance and data marketplaces",
    "the robotics foundation model revolution requiring massive manipulation datasets",
    "the synthetic data boom and its impact on model quality and bias",
    "the rise of mixture-of-experts architectures changing inference economics",
    "the developer tools ecosystem shifting toward AI-native workflows",
    "the enterprise AI adoption wave creating demand for domain-specific fine-tuning data",
    "autonomous systems requiring real-time edge inference with safety guarantees",
    "the multimodal AI revolution combining vision, language, and structured data",
    "the open-source AI ecosystem's rapid catch-up to frontier closed models",
]

# ═══════════════════════════════════════════════════════════════════════════════
# COOK LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_stats = {
    "total_written": 0,
    "api_calls": 0,
    "gen_pass": 0,
    "rewritten": 0,
    "errors": 0,
    "categories": {k: {"written": 0, "target": v} for k, v in TARGETS.items()},
}


def _api(messages, max_tokens, temperature=0.85):
    key = os.environ.get("TOGETHER_KEY") or os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("Set TOGETHER_KEY or TOGETHER_API_KEY")
    resp = requests.post(
        TOGETHER_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": GEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        },
        timeout=120,
    )
    with _lock:
        _stats["api_calls"] += 1
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _rewrite(text, fmt_spec):
    """Quality rewrite pass using same model."""
    messages = [
        {"role": "system", "content": "You are a senior AI infrastructure analyst. Rewrite the following to be more precise, technical, and structured. Keep the EXACT same format headers. Make every sentence count."},
        {"role": "user", "content": text},
    ]
    return _api(messages, fmt_spec["max_tokens"], temperature=0.7)


def _quality_check(text, fmt_name, fmt_spec):
    """Check if output meets quality bar."""
    if len(text) < fmt_spec["min_len"]:
        return False

    if fmt_name == "SIGNAL_EXTRACTION":
        return all(h in text for h in ["SIGNAL:", "WHY IT MATTERS:", "DATA OPPORTUNITY:", "CURATION SCORE:"])
    elif fmt_name == "GAP_DETECTION":
        return all(h in text for h in ["CURRENT CAPABILITIES:", "KNOWN WEAKNESSES:", "DATA GAP:", "CURATION OBJECT:"])
    elif fmt_name == "TREND_SYNTHESIS":
        return all(h in text for h in ["NAME:", "DESCRIPTION:", "WHY NOW:", "TARGET PAIRS:", "EXPECTED MODEL IMPACT:"])
    return True


def generate_signal_extraction():
    topic = random.choice(SIGNAL_TOPICS)
    fmt = FORMATS["SIGNAL_EXTRACTION"]
    messages = [
        {"role": "system", "content": "You are a senior AI signal analyst for a data intelligence company. You identify weak signals in AI infrastructure developments that indicate dataset curation opportunities. Be specific, quantitative, and actionable."},
        {"role": "user", "content": f"{fmt['instruction']}\n\nDevelopment to analyze:\n{topic}"},
    ]
    return messages, "SIGNAL_EXTRACTION", fmt


def generate_gap_detection():
    domain = random.choice(GAP_DOMAINS)
    fmt = FORMATS["GAP_DETECTION"]
    messages = [
        {"role": "system", "content": "You are a senior AI model evaluator for a data intelligence company. You analyze model capabilities and identify specific data gaps that prevent models from performing well. Be precise about what training data is missing and why."},
        {"role": "user", "content": f"{fmt['instruction']}\n\nDomain to analyze:\n{domain}"},
    ]
    return messages, "GAP_DETECTION", fmt


def generate_trend_synthesis():
    context = random.choice(TREND_CONTEXTS)
    fmt = FORMATS["TREND_SYNTHESIS"]
    messages = [
        {"role": "system", "content": "You are the chief data curator for an AI intelligence company. You synthesize market signals into actionable dataset curation plans. Your Top-5 lists drive what data gets built next. Be specific about pair counts, formats, and expected impact."},
        {"role": "user", "content": f"{fmt['instruction']}\n\nCurrent market context:\n{context}"},
    ]
    return messages, "TREND_SYNTHESIS", fmt


GENERATORS = {
    "signal_extraction": generate_signal_extraction,
    "gap_detection": generate_gap_detection,
    "trend_synthesis": generate_trend_synthesis,
}


def cook_one(category):
    """Cook one pair for the given category."""
    gen_fn = GENERATORS[category]
    messages, fmt_name, fmt_spec = gen_fn()

    try:
        text = _api(messages, fmt_spec["max_tokens"])

        if not _quality_check(text, fmt_name, fmt_spec):
            # Rewrite pass
            text = _rewrite(text, fmt_spec)
            with _lock:
                _stats["rewritten"] += 1
            if not _quality_check(text, fmt_name, fmt_spec):
                with _lock:
                    _stats["errors"] += 1
                return None
        else:
            with _lock:
                _stats["gen_pass"] += 1

        pair = {
            "messages": [
                messages[0],  # system
                messages[1],  # user
                {"role": "assistant", "content": text},
            ],
            "format": fmt_name,
            "category": category,
            "cook": "cascade_v1",
        }
        return pair

    except Exception as e:
        with _lock:
            _stats["errors"] += 1
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open output files
    files = {}
    for cat in TARGETS:
        files[cat] = open(OUT_DIR / f"cascade_{cat}.jsonl", "a")

    # Load existing progress
    for cat in TARGETS:
        path = OUT_DIR / f"cascade_{cat}.jsonl"
        if path.exists():
            with open(path) as f:
                count = sum(1 for l in f if l.strip())
            _stats["categories"][cat]["written"] = count
            _stats["total_written"] += count
            print(f"  Resuming {cat}: {count}/{TARGETS[cat]}")

    total_target = sum(TARGETS.values())
    print(f"\n{'='*60}")
    print(f"SwarmSignal Cascade Cook — {total_target:,} Intelligence Pairs")
    print(f"{'='*60}")
    for cat, target in TARGETS.items():
        print(f"  {cat}: {target:,}")
    print(f"  Model: {GEN_MODEL}")
    print(f"{'='*60}\n")

    start = time.time()

    def _worker():
        while True:
            # Pick category that needs more pairs
            with _lock:
                candidates = [
                    (cat, TARGETS[cat] - _stats["categories"][cat]["written"])
                    for cat in TARGETS
                    if _stats["categories"][cat]["written"] < TARGETS[cat]
                ]
            if not candidates:
                return

            # Weight by remaining
            total_remaining = sum(r for _, r in candidates)
            if total_remaining <= 0:
                return

            roll = random.random() * total_remaining
            cumulative = 0
            category = candidates[0][0]
            for cat, remaining in candidates:
                cumulative += remaining
                if roll < cumulative:
                    category = cat
                    break

            pair = cook_one(category)
            if pair:
                with _lock:
                    _stats["categories"][category]["written"] += 1
                    _stats["total_written"] += 1
                    files[category].write(json.dumps(pair) + "\n")
                    files[category].flush()

                    # Progress
                    w = _stats["total_written"]
                    if w % 50 == 0:
                        elapsed = time.time() - start
                        rate = w / (elapsed / 60) if elapsed > 0 else 0
                        remaining = total_target - w
                        eta_min = remaining / rate if rate > 0 else 0
                        print(f"  [{w:,}/{total_target:,}] {rate:.0f}/min | "
                              f"ETA {eta_min:.0f}m | "
                              f"SE:{_stats['categories']['signal_extraction']['written']} "
                              f"GD:{_stats['categories']['gap_detection']['written']} "
                              f"TS:{_stats['categories']['trend_synthesis']['written']}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_worker) for _ in range(WORKERS)]
        for f in futures:
            f.result()

    # Close files
    for f in files.values():
        f.close()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Cascade Cook DONE — {_stats['total_written']:,} pairs in {elapsed/60:.1f} min")
    print(f"  API calls:  {_stats['api_calls']:,}")
    print(f"  Gen pass:   {_stats['gen_pass']:,}")
    print(f"  Rewritten:  {_stats['rewritten']:,}")
    print(f"  Errors:     {_stats['errors']:,}")
    for cat in TARGETS:
        s = _stats["categories"][cat]
        print(f"  {cat}: {s['written']}/{s['target']}")
    print(f"{'='*60}")

    # Save progress
    with open(OUT_DIR / "progress.json", "w") as f:
        json.dump({**_stats, "elapsed_min": elapsed / 60, "completed": datetime.now(timezone.utc).isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
