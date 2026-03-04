#!/usr/bin/env python3
"""SwarmSignal Daily Report Generator
=======================================

Generates daily briefings and weekly journals using the fine-tuned
SwarmSignal-9B model served via llama-server or Ollama.

Flow: collect_signals.py output → format context → SwarmSignal model → report

Usage:
    # Daily briefing (default)
    python3 -m src.generate_daily

    # Weekly journal
    python3 -m src.generate_daily --format journal

    # Custom signals file
    python3 -m src.generate_daily --signals /path/to/signals.json

    # Use Ollama instead of llama-server
    python3 -m src.generate_daily --backend ollama

    # Dry run — show prompt without generating
    python3 -m src.generate_daily --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SIGNALS_DIR = Path("/data2/swarmsignal/signals")
REPORTS_DIR = Path("/data2/swarmsignal/reports")

# Model serving backends
BACKENDS = {
    "llama": {
        "url": "http://localhost:8081/v1/chat/completions",
        "model": "swarmsignal-9b",
    },
    "ollama": {
        "url": "http://localhost:11434/api/chat",
        "model": "swarmsignal-9b",
    },
}

SYSTEM_PROMPT = """You are SwarmSignal, an AI market intelligence analyst and writer for Swarm & Bee.

Your job is to produce sharp, insightful analysis of AI trends, products, culture, and market dynamics. You write with clarity, conviction, and edge — no corporate fluff. You connect dots others miss. You call out hype and highlight substance.

Voice guidelines:
- Write like a senior analyst with opinions, not a press release bot
- Lead with insight, not summary — tell the reader what it MEANS
- Use concrete specifics: names, numbers, dates, model sizes, funding amounts
- When you're bullish, say why with evidence. When you're skeptical, same.
- Cultural commentary should be real — capture what devs actually talk about
- No "leveraging", "synergies", "paradigm shift", "game-changing", "revolutionary"
- Short punchy sentences mixed with longer analytical ones
- You can be funny, wry, even sarcastic when warranted — you're not a bot

Output formats:
- BRIEFING: Concise daily intel — 5-7 top stories with 2-3 sentence analysis each, overall sentiment score (1-10 bearish→bullish), 2-3 trend callouts, one "sleeper story" others are missing. ~800-1200 words.
- JOURNAL: Long-form weekly synthesis — market pulse overview, 2-3 deep dives on the week's biggest themes, vibe check (what the community is feeling), contrarian take, predictions. ~2000-3000 words."""


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def load_signals(path: Path | None = None) -> dict:
    """Load today's signals file."""
    if path and path.exists():
        return json.loads(path.read_text())

    # Find latest signals file
    if SIGNALS_DIR.exists():
        files = sorted(SIGNALS_DIR.glob("signals_*.json"), reverse=True)
        if files:
            return json.loads(files[0].read_text())

    return None


def format_signals_for_briefing(signals: dict) -> str:
    """Format collected signals into a briefing prompt."""
    date = signals.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    items = signals.get("signals", [])

    # Group by source
    by_source = {}
    for item in items:
        src = item.get("source", "unknown")
        by_source.setdefault(src, []).append(item)

    lines = [
        f"Write a SwarmSignal daily BRIEFING for {date}.",
        "",
        f"Today's raw signals ({len(items)} total from {len(by_source)} sources):",
        "",
    ]

    # Top stories from news sources
    news_items = [i for i in items if i["source"] != "arxiv"]
    if news_items:
        lines.append("## News & Industry")
        for item in news_items[:15]:
            score_str = ""
            if item.get("score"):
                score_str = f" [HN: {item['score']} pts, {item.get('comments', 0)} comments]"
            lines.append(f"- {item['title']}{score_str}")
            if item.get("summary"):
                lines.append(f"  {item['summary'][:200]}")
        lines.append("")

    # ArXiv papers
    arxiv_items = [i for i in items if i["source"] == "arxiv"]
    if arxiv_items:
        lines.append("## Research Papers (arXiv)")
        for item in arxiv_items[:10]:
            authors = ", ".join(item.get("authors", [])[:3])
            if authors:
                authors = f" ({authors})"
            lines.append(f"- {item['title']}{authors}")
            if item.get("summary"):
                lines.append(f"  {item['summary'][:200]}")
        lines.append("")

    lines.append("Analyze these signals. Include: sentiment score (1-10), trend callouts, sleeper story.")
    return "\n".join(lines)


def format_signals_for_journal(signals: dict) -> str:
    """Format collected signals into a weekly journal prompt."""
    date = signals.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    items = signals.get("signals", [])

    lines = [
        f"Write a SwarmSignal weekly JOURNAL for the week ending {date}.",
        "",
        f"This week's signal feed ({len(items)} items):",
        "",
    ]

    for item in items[:30]:
        lines.append(f"- [{item.get('source', '?')}] {item['title']}")
        if item.get("summary"):
            lines.append(f"  {item['summary'][:150]}")

    lines.append("")
    lines.append(
        "Write a full weekly journal with: market pulse overview, "
        "2-3 deep dives on the biggest themes, vibe check, "
        "one contrarian take, and 3 predictions for next week. "
        "2000-3000 words of real analysis."
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_llama(prompt: str, config: dict) -> str:
    """Generate via llama-server (OpenAI-compatible API)."""
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    resp = requests.post(config["url"], json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_ollama(prompt: str, config: dict) -> str:
    """Generate via Ollama API."""
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 4096,
        },
    }

    resp = requests.post(config["url"], json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SwarmSignal Daily Report Generator")
    parser.add_argument("--format", choices=["briefing", "journal"], default="briefing")
    parser.add_argument("--signals", type=Path, default=None, help="Path to signals JSON")
    parser.add_argument("--backend", choices=["llama", "ollama"], default="llama")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt, don't generate")
    args = parser.parse_args()

    # Load signals
    signals = load_signals(args.signals)
    if not signals:
        print("ERROR: No signals found. Run collect_signals.py first.")
        sys.exit(1)

    date = signals.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    print(f"SwarmSignal Report Generator — {date}")
    print(f"  Format: {args.format}")
    print(f"  Signals: {signals.get('total_signals', 0)} items")
    print(f"  Backend: {args.backend}")

    # Format prompt
    if args.format == "briefing":
        prompt = format_signals_for_briefing(signals)
    else:
        prompt = format_signals_for_journal(signals)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"PROMPT ({len(prompt)} chars):")
        print(f"{'='*60}")
        print(prompt)
        return

    # Generate
    config = BACKENDS[args.backend]
    print(f"\n  Generating with {config['model']}...")

    try:
        if args.backend == "ollama":
            report = generate_ollama(prompt, config)
        else:
            report = generate_llama(prompt, config)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {config['url']}")
        print(f"  Start the model server first:")
        if args.backend == "llama":
            print(f"  llama-server -m swarmsignal-9b-q4km.gguf -ngl 99 --port 8081")
        else:
            print(f"  ollama run swarmsignal-9b")
        sys.exit(1)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or REPORTS_DIR / f"{args.format}_{date}.md"

    header = f"# SwarmSignal {'Daily Briefing' if args.format == 'briefing' else 'Weekly Journal'}\n"
    header += f"**Date:** {date}\n"
    header += f"**Signals:** {signals.get('total_signals', 0)} items from {len(signals.get('sources', {}))} sources\n\n"
    header += "---\n\n"

    with open(output_path, "w") as f:
        f.write(header + report)

    print(f"\n  Report saved: {output_path}")
    print(f"  Length: {len(report):,} chars")

    # Print preview
    preview = report[:500]
    print(f"\n{'='*60}")
    print(f"PREVIEW:")
    print(f"{'='*60}")
    print(preview)
    if len(report) > 500:
        print(f"\n  ... ({len(report) - 500:,} more chars)")


if __name__ == "__main__":
    main()
