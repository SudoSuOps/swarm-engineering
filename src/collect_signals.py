#!/usr/bin/env python3
"""SwarmSignal Daily News Collector
====================================

Lightweight RSS + API collector for AI news signals.
Feeds context to SwarmSignal model for daily report generation.

Sources (all free, no auth required):
  - RSS: TechCrunch AI, The Verge AI, Ars Technica, MIT Tech Review, VentureBeat
  - ArXiv: cs.AI, cs.CL, cs.LG (latest papers via API)
  - Hacker News: Top stories filtered for AI keywords

Output: /data2/swarmsignal/signals/signals_YYYY-MM-DD.json

Usage:
    python3 -m src.collect_signals
    python3 -m src.collect_signals --output /path/to/output.json
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

OUTPUT_DIR = Path("/data2/swarmsignal/signals")

# ═══════════════════════════════════════════════════════════════════════════════
# RSS FEEDS
# ═══════════════════════════════════════════════════════════════════════════════

RSS_FEEDS = {
    "techcrunch_ai": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "verge_ai": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "ars_technica": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "mit_tech_review": "https://www.technologyreview.com/feed/",
    "venturebeat_ai": "https://venturebeat.com/category/ai/feed/",
}

# AI keywords for filtering general feeds
AI_KEYWORDS = re.compile(
    r"\b(artificial intelligence|machine learning|deep learning|neural network|"
    r"large language model|LLM|GPT|Claude|Gemini|transformer|diffusion|"
    r"generative AI|gen ?AI|AI agent|AI model|fine[- ]tun|RLHF|"
    r"OpenAI|Anthropic|DeepMind|Meta AI|Mistral|Hugging Face|"
    r"GPU|NVIDIA|H100|B200|TPU|CUDA|inference|training run|"
    r"open source model|chat ?bot|copilot|AI safety|alignment|"
    r"multimodal|vision model|reasoning model|benchmark|"
    r"tokenizer|embedding|vector|RAG|retrieval|prompt)\b",
    re.IGNORECASE,
)


def fetch_rss(name: str, url: str, limit: int = 20) -> list[dict]:
    """Fetch and parse RSS feed, return structured items."""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "SwarmSignal/1.0 (AI News Collector)"
        })
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARN: {name} fetch failed: {e}")
        return []

    items = []
    try:
        root = ET.fromstring(resp.text)

        # Handle both RSS 2.0 and Atom feeds
        # RSS 2.0: channel/item
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()

            # Strip HTML from description
            desc = re.sub(r"<[^>]+>", "", desc).strip()[:500]

            items.append({
                "source": name,
                "title": title,
                "url": link,
                "summary": desc,
                "published": pub_date,
            })

        # Atom: entry
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns)[:limit]:
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
            pub_date = (entry.findtext("atom:published", namespaces=ns) or
                        entry.findtext("atom:updated", namespaces=ns) or "").strip()

            summary = re.sub(r"<[^>]+>", "", summary).strip()[:500]

            if title and title not in [i["title"] for i in items]:
                items.append({
                    "source": name,
                    "title": title,
                    "url": link,
                    "summary": summary,
                    "published": pub_date,
                })

    except ET.ParseError as e:
        print(f"  WARN: {name} XML parse error: {e}")

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# ARXIV
# ═══════════════════════════════════════════════════════════════════════════════

ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]
ARXIV_API = "http://export.arxiv.org/api/query"


def fetch_arxiv(limit: int = 20) -> list[dict]:
    """Fetch latest AI papers from arXiv API."""
    cats = "+OR+".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    params = {
        "search_query": cats,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": limit,
    }

    try:
        resp = requests.get(ARXIV_API, params=params, timeout=20, headers={
            "User-Agent": "SwarmSignal/1.0"
        })
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARN: arXiv fetch failed: {e}")
        return []

    items = []
    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)

        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
            title = re.sub(r"\s+", " ", title)  # collapse whitespace

            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
            summary = re.sub(r"\s+", " ", summary)[:500]

            link = ""
            for link_el in entry.findall("atom:link", ns):
                if link_el.get("type") == "text/html":
                    link = link_el.get("href", "")
                    break
            if not link:
                link_el = entry.find("atom:id", ns)
                link = link_el.text if link_el is not None else ""

            pub_date = (entry.findtext("atom:published", namespaces=ns) or "").strip()

            # Authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", namespaces=ns)
                if name:
                    authors.append(name.strip())

            # Categories
            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            items.append({
                "source": "arxiv",
                "title": title,
                "url": link,
                "summary": summary,
                "published": pub_date,
                "authors": authors[:5],
                "categories": categories,
            })

    except ET.ParseError as e:
        print(f"  WARN: arXiv XML parse error: {e}")

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# HACKER NEWS
# ═══════════════════════════════════════════════════════════════════════════════

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"


def fetch_hackernews(limit: int = 15) -> list[dict]:
    """Fetch top HN stories, filter for AI-related."""
    try:
        resp = requests.get(HN_TOP, timeout=10)
        resp.raise_for_status()
        story_ids = resp.json()[:100]  # Check top 100 for AI stories
    except Exception as e:
        print(f"  WARN: HN fetch failed: {e}")
        return []

    items = []
    for sid in story_ids:
        if len(items) >= limit:
            break
        try:
            resp = requests.get(HN_ITEM.format(sid), timeout=5)
            story = resp.json()
            if not story or story.get("type") != "story":
                continue

            title = story.get("title", "")
            url = story.get("url", f"https://news.ycombinator.com/item?id={sid}")

            # Filter for AI-related
            if not AI_KEYWORDS.search(title):
                continue

            items.append({
                "source": "hackernews",
                "title": title,
                "url": url,
                "summary": "",
                "published": datetime.fromtimestamp(
                    story.get("time", 0), tz=timezone.utc
                ).isoformat(),
                "score": story.get("score", 0),
                "comments": story.get("descendants", 0),
            })
        except Exception:
            continue

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def collect(output_path: Path | None = None) -> dict:
    """Collect all signals and save to JSON."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"signals_{today}.json"

    print(f"SwarmSignal Collector — {today}")
    print(f"{'='*50}")

    all_items = []

    # RSS feeds
    for name, url in RSS_FEEDS.items():
        print(f"  Fetching {name}...")
        items = fetch_rss(name, url)
        # Filter general feeds for AI content
        if name not in ("techcrunch_ai", "verge_ai", "venturebeat_ai"):
            items = [i for i in items if AI_KEYWORDS.search(
                f"{i['title']} {i['summary']}"
            )]
        all_items.extend(items)
        print(f"    -> {len(items)} stories")

    # ArXiv
    print(f"  Fetching arXiv (cs.AI, cs.CL, cs.LG)...")
    arxiv_items = fetch_arxiv()
    all_items.extend(arxiv_items)
    print(f"    -> {len(arxiv_items)} papers")

    # Hacker News
    print(f"  Fetching Hacker News (AI filter)...")
    hn_items = fetch_hackernews()
    all_items.extend(hn_items)
    print(f"    -> {len(hn_items)} stories")

    # Dedup by title similarity
    seen_titles = set()
    unique_items = []
    for item in all_items:
        title_key = re.sub(r"[^a-z0-9]", "", item["title"].lower())[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_items.append(item)

    # Build output
    result = {
        "date": today,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_signals": len(unique_items),
        "sources": {
            name: len([i for i in unique_items if i["source"] == name])
            for name in set(i["source"] for i in unique_items)
        },
        "signals": unique_items,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Total: {len(unique_items)} unique signals")
    print(f"  Output: {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmSignal Daily News Collector")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    collect(output_path=args.output)
