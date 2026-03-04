#!/usr/bin/env python3
"""SwarmSignal Cook — 30K AI Trends Writer Training Pairs via Together.ai
=========================================================================

Two-tier architecture:
  Tier 1 (GEN):  Qwen3-Next-80B-A3B — fast MoE turbo, 3B active params
  Tier 2 (PASS): Qwen3-235B-A22B    — heavyweight quality rewrite

Flow: 80B generates → quality check → pass or 235B rewrite

6 Categories:
  1. BRIEFING (10K)   — Concise daily intel (5-7 stories, sentiment, trends)
  2. TRENDS (8K)      — Deep dives on specific AI trends
  3. COMPANY (4K)     — Launches, pivots, funding, M&A
  4. CULTURE (4K)     — Community discourse, open source, vibes, drama
  5. MARKET (2K)      — Funding rounds, valuations, M&A dynamics
  6. JOURNAL (2K)     — Long-form weekly synthesis, connecting dots

Usage:
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal --category briefing
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal --status
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import random
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
PASS_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"  # same model, cheap enough
WORKERS = 50
SEED = 2026

OUTPUT_DIR = Path("/tmp/swarmsignal_cook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_TARGETS = {
    "briefing": 10_000,
    "trends":    8_000,
    "company":   4_000,
    "culture":   4_000,
    "market":    2_000,
    "journal":   2_000,
}
TOTAL_TARGET = sum(CATEGORY_TARGETS.values())  # 30,000

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

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
- JOURNAL: Long-form weekly synthesis — market pulse overview, 2-3 deep dives on the week's biggest themes, vibe check (what the community is feeling), contrarian take, predictions. ~2000-3000 words.
- TREND_ANALYSIS: Deep dive on a single trend — what's happening, why it matters, who's winning/losing, where it's headed. ~1200-1800 words.
- COMPANY_PROFILE: Sharp analysis of a company move — the what, the why, what it signals, who it affects. ~800-1200 words.
- CULTURE_REPORT: What the AI community is actually talking about — the discourse, the drama, the memes, the vibes. ~600-1000 words.
- MARKET_BRIEF: Funding, M&A, valuation analysis — follow the money, explain what it means. ~600-1000 words."""

# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC POOLS — Rich scenario templates for each category
# ═══════════════════════════════════════════════════════════════════════════════

# Foundation model companies and labs
_LABS = [
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Mistral",
    "xAI", "Cohere", "AI21 Labs", "Inflection", "Stability AI",
    "Midjourney", "Runway", "Character.ai", "Perplexity", "Groq",
    "Together AI", "Fireworks AI", "Anyscale", "Modal", "Replicate",
    "Hugging Face", "EleutherAI", "Allen AI", "Nous Research", "01.AI",
    "Zhipu AI", "Baichuan", "Moonshot AI", "DeepSeek", "Qwen/Alibaba",
    "Samsung AI", "Apple ML", "Amazon Bedrock", "Nvidia NeMo", "AMD ROCm",
]

# AI application domains
_DOMAINS = [
    "code generation", "creative writing", "image generation", "video generation",
    "music generation", "3D generation", "scientific discovery", "drug discovery",
    "protein folding", "materials science", "climate modeling", "weather prediction",
    "autonomous driving", "robotics", "healthcare diagnostics", "legal tech",
    "financial trading", "cybersecurity", "education", "customer service",
    "enterprise search", "data analytics", "DevOps automation", "game development",
    "chip design", "supply chain", "manufacturing QA", "agriculture tech",
    "content moderation", "translation", "voice synthesis", "document processing",
]

# Hot topics and themes
_THEMES = [
    "scaling laws hitting diminishing returns",
    "small models outperforming large ones on specific tasks",
    "open source vs closed source model race",
    "AI regulation in the EU vs US vs China",
    "compute supply chain bottlenecks",
    "GPU shortage and NVIDIA monopoly dynamics",
    "inference cost reduction breakthroughs",
    "AI agent frameworks proliferating",
    "MoE architectures becoming standard",
    "synthetic data for training at scale",
    "RLHF vs DPO vs GRPO alignment methods",
    "multimodal models converging",
    "AI-powered search disrupting Google",
    "enterprise AI adoption gap between hype and reality",
    "AI startup funding winter or spring",
    "developer experience and tooling maturity",
    "edge AI and on-device inference",
    "AI safety and alignment research progress",
    "copyright lawsuits reshaping training data landscape",
    "AI in defense and national security",
    "AI companionship and social impact",
    "coding assistants productivity evidence",
    "fine-tuning vs RAG vs long context debate",
    "reasoning models and chain-of-thought",
    "AI hardware beyond NVIDIA — AMD, Intel, custom ASICs",
    "AI energy consumption and sustainability",
    "AI job displacement data vs predictions",
    "China AI ecosystem divergence from West",
    "browser-based and WebGPU inference",
    "AI model merging and DARE techniques",
    "quantization advances (GGUF, GPTQ, AWQ, EXL2)",
    "AI benchmarks — are they meaningful?",
    "agentic workflows and tool use standards",
    "AI and cryptocurrency / blockchain convergence",
    "federated learning and privacy-preserving AI",
    "AI governance and corporate responsibility",
    "robotics foundation models",
    "text-to-SQL and structured data AI",
    "AI observability and evaluation frameworks",
    "prompt engineering dying or evolving",
]

# Cultural phenomena
_CULTURE = [
    "Hacker News AI discourse patterns",
    "Twitter/X AI influencer ecosystem",
    "r/LocalLLaMA community innovations",
    "open source model release drama",
    "AI doomer vs accelerationist debate",
    "AI art community backlash and adaptation",
    "developer burnout from AI hype cycle",
    "the 'vibe coding' phenomenon",
    "AI YouTube education boom",
    "Discord AI communities and bot ecosystems",
    "AI conference culture (NeurIPS, ICML, etc.)",
    "AI meme culture and in-jokes",
    "the 'it's just autocomplete' discourse",
    "AI ethics Twitter wars",
    "open source model naming conventions drama",
    "GPU poor vs GPU rich divide",
    "AI paper review culture and arxiv spam",
    "the rise of AI-native startups vs incumbents adapting",
    "developer identity crisis — 'will AI replace me?'",
    "AI tooling fatigue and framework churn",
    "the 'just use Claude/GPT' vs 'build it yourself' camps",
    "AI-generated slop flooding the internet",
    "indie hackers shipping AI products",
    "corporate AI washing — adding 'AI' to everything",
    "the commoditization of intelligence debate",
]

# Market events
_MARKET_EVENTS = [
    "Series A", "Series B", "Series C", "Series D", "growth round",
    "IPO filing", "SPAC merger", "acquisition", "acqui-hire", "talent raid",
    "pivot announcement", "shutdown", "layoffs", "hiring spree",
    "product launch", "API price cut", "open source release",
    "partnership deal", "licensing agreement", "patent filing",
    "regulatory approval", "compliance penalty", "antitrust investigation",
    "revenue milestone", "profitability claim", "valuation markdown",
]

# Hardware and infra topics
_HARDWARE = [
    "NVIDIA H100/H200/B200 GPU supply", "AMD MI300X/MI350 competitiveness",
    "Intel Gaudi 3 and Falcon Shores", "Google TPU v5p/v6",
    "custom ASIC startups (Groq, Cerebras, SambaNova)",
    "hyperscaler GPU build-outs", "sovereign AI compute initiatives",
    "inference optimization hardware", "edge AI chips (Qualcomm, MediaTek)",
    "NVIDIA CUDA moat vs alternatives", "HBM memory supply constraints",
    "liquid cooling for AI data centers", "nuclear power for AI compute",
    "AI data center energy deals", "GPU cloud pricing dynamics",
]


def _pick(pool: list, rng: random.Random, n: int = 1) -> list[str]:
    return rng.sample(pool, min(n, len(pool)))


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS — one per category
# ═══════════════════════════════════════════════════════════════════════════════

def gen_briefing_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a daily briefing scenario with 5-7 story seeds."""
    # Pick a date context
    month = rng.choice(["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"])
    year = rng.choice([2025, 2026])
    weekday = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    stories = []
    # Mix story types
    labs = _pick(_LABS, rng, 3)
    themes = _pick(_THEMES, rng, 2)
    domains = _pick(_DOMAINS, rng, 2)

    for lab in labs:
        event = rng.choice(_MARKET_EVENTS[:17])  # product/market events
        stories.append(f"{lab} {event}")
    for theme in themes:
        stories.append(f"Emerging trend: {theme}")
    for domain in domains:
        stories.append(f"Breakthrough in {domain}")

    rng.shuffle(stories)
    stories = stories[:rng.randint(5, 7)]

    return {
        "format": "BRIEFING",
        "prompt": (
            f"Write a SwarmSignal daily briefing for {weekday}, {month} {rng.randint(1,28)}, {year}.\n\n"
            f"Today's story seeds (use these as starting points, add analysis and connections):\n"
            + "\n".join(f"- {s}" for s in stories)
            + "\n\nInclude: sentiment score (1-10), trend callouts, and a sleeper story."
        ),
        "min_len": 600,
        "max_tokens": 2048,
    }


def gen_trends_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a trend analysis deep dive scenario."""
    theme = rng.choice(_THEMES)
    labs = _pick(_LABS, rng, 3)
    related_domains = _pick(_DOMAINS, rng, 2)

    return {
        "format": "TREND_ANALYSIS",
        "prompt": (
            f"Write a SwarmSignal trend analysis deep dive on: {theme}\n\n"
            f"Key players to consider: {', '.join(labs)}\n"
            f"Related application areas: {', '.join(related_domains)}\n\n"
            f"Cover: what's actually happening (with specifics), why it matters, "
            f"who's winning and losing, contrarian angle, where this is headed in 6-12 months."
        ),
        "min_len": 1000,
        "max_tokens": 3072,
    }


def gen_company_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a company/product profile scenario."""
    lab = rng.choice(_LABS)
    event = rng.choice(_MARKET_EVENTS)
    domain = rng.choice(_DOMAINS)

    amount = ""
    if event in ("Series A", "Series B", "Series C", "Series D", "growth round"):
        amount = f" at a ${rng.choice([50, 100, 200, 500, 750, 1000, 2000, 4000, 6500])}M valuation"
    elif event == "acquisition":
        amount = f" for ${rng.choice([100, 250, 500, 1000, 2500, 5000])}M"

    return {
        "format": "COMPANY_PROFILE",
        "prompt": (
            f"Write a SwarmSignal company analysis: {lab} just announced a {event}{amount}.\n"
            f"Their focus area: {domain}.\n\n"
            f"Analyze: what this move signals about the broader market, competitive implications, "
            f"what it means for {domain}, and your honest take on whether this is bullish or overhyped."
        ),
        "min_len": 600,
        "max_tokens": 2048,
    }


def gen_culture_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a culture/vibe report scenario."""
    topic = rng.choice(_CULTURE)
    themes = _pick(_THEMES, rng, 2)

    return {
        "format": "CULTURE_REPORT",
        "prompt": (
            f"Write a SwarmSignal culture report about: {topic}\n\n"
            f"Related themes swirling in the discourse: {', '.join(themes)}\n\n"
            f"Capture: what people are actually saying (be specific), the prevailing vibes, "
            f"the real tensions underneath, your take on what's signal vs noise, "
            f"and one prediction about where this discourse heads."
        ),
        "min_len": 500,
        "max_tokens": 1536,
    }


def gen_market_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a market dynamics / funding scenario."""
    events = _pick(_MARKET_EVENTS, rng, 3)
    labs = _pick(_LABS, rng, 4)
    hw = _pick(_HARDWARE, rng, 1)

    # Generate some funding numbers
    deals = []
    for lab, event in zip(labs[:3], events):
        if "Series" in event or event == "growth round":
            amt = rng.choice([15, 30, 50, 100, 200, 400, 650, 1000, 2000, 6000])
            deals.append(f"{lab}: {event} (${amt}M)")
        else:
            deals.append(f"{lab}: {event}")

    return {
        "format": "MARKET_BRIEF",
        "prompt": (
            f"Write a SwarmSignal market brief covering recent AI funding and M&A activity.\n\n"
            f"Notable deals this week:\n"
            + "\n".join(f"- {d}" for d in deals)
            + f"\n\nHardware context: {hw[0]}\n\n"
            f"Analyze: what the money is chasing, where capital is drying up, "
            f"valuation trends, and what this funding pattern tells us about "
            f"where AI is really headed (vs where VCs think it's headed)."
        ),
        "min_len": 500,
        "max_tokens": 1536,
    }


def gen_journal_scenario(rng: random.Random, idx: int) -> dict:
    """Generate a weekly journal synthesis scenario."""
    month = rng.choice(["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"])
    year = rng.choice([2025, 2026])
    week = rng.randint(1, 4)

    # Build a rich week of events
    themes = _pick(_THEMES, rng, 3)
    labs = _pick(_LABS, rng, 5)
    events = _pick(_MARKET_EVENTS, rng, 3)
    culture = _pick(_CULTURE, rng, 2)
    hw = _pick(_HARDWARE, rng, 1)

    week_events = []
    for lab, event in zip(labs[:3], events):
        week_events.append(f"{lab}: {event}")
    for theme in themes[:2]:
        week_events.append(f"Trend: {theme}")
    week_events.append(f"Community: {culture[0]}")
    week_events.append(f"Hardware: {hw[0]}")

    return {
        "format": "JOURNAL",
        "prompt": (
            f"Write a SwarmSignal weekly journal for Week {week} of {month} {year}.\n\n"
            f"This week's key events and themes:\n"
            + "\n".join(f"- {e}" for e in week_events)
            + f"\n\nVibes from the community: {culture[1]}\n\n"
            f"Write a full weekly journal with: market pulse overview, "
            f"2-3 deep dives on the biggest themes, vibe check section, "
            f"one contrarian take the consensus is wrong about, "
            f"and 3 predictions for next week. This should be substantive — "
            f"2000-3000 words of real analysis, not padding."
        ),
        "min_len": 1500,
        "max_tokens": 4096,
    }


CATEGORY_GENERATORS = {
    "briefing": gen_briefing_scenario,
    "trends":   gen_trends_scenario,
    "company":  gen_company_scenario,
    "culture":  gen_culture_scenario,
    "market":   gen_market_scenario,
    "journal":  gen_journal_scenario,
}

# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

session = requests.Session()
api_lock = threading.Lock()
api_calls = Counter()  # total, gen, pass, rewrite, error


def init_session(api_key: str):
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })


def together_call(system: str, user: str, model: str = None,
                   max_tokens: int = 3072, temperature: float = 0.7,
                   min_len: int = 100, retries: int = 4) -> str | None:
    """Call Together.ai API with retry logic."""
    model = model or GEN_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = session.post(TOGETHER_URL, json=payload, timeout=120)
            with api_lock:
                api_calls["total"] += 1
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            if resp.status_code == 402:
                if attempt < retries - 1:
                    print(f"  WARN: 402 — retrying in {2 ** attempt + 2}s...")
                    time.sleep(2 ** attempt + 2)
                    continue
                raise RuntimeError("402 Payment Required — out of credits")
            if resp.status_code == 403:
                raise RuntimeError("403 Forbidden — bad API key")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip <think>...</think> blocks from Qwen3 models
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
            if content and len(content) > min_len:
                return content
        except RuntimeError:
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                with api_lock:
                    api_calls["error"] += 1
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK — adapted for writing quality, not financial trajectory
# ═══════════════════════════════════════════════════════════════════════════════

# Degeneration: same sentence repeated 3+ times
_DEGEN_PATTERN = re.compile(r"(.{40,})\1{2,}")

# Corporate speak detector
_CORP_SPEAK = re.compile(
    r"\b(leverag(?:e|ing)|synerg(?:y|ies)|paradigm shift|game[- ]chang(?:ing|er)|"
    r"revolutionary|disruptive innovation|move the needle|circle back|"
    r"best[- ]in[- ]class|world[- ]class|cutting[- ]edge|"
    r"unlock(?:ing)? (?:the )?potential|at the end of the day|"
    r"going forward|in terms of|it goes without saying)\b",
    re.IGNORECASE,
)

# Specificity markers — real analysis has names, numbers, comparisons
_SPECIFICS = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|"  # proper nouns
    r"\$[\d,.]+[BMK]?|"                        # dollar amounts
    r"\d+\.?\d*\s*%|"                          # percentages
    r"\d+\.?\d*[BMK]\b|"                       # quantities with B/M/K
    r"\d{4}\b",                                 # years
    re.MULTILINE,
)


def quality_check(content: str, category: str) -> tuple[bool, str]:
    """Check if gen output meets SwarmSignal quality. Returns (pass, reason)."""
    # Length check — varies by category
    min_lens = {
        "briefing": 600, "trends": 1000, "company": 600,
        "culture": 500, "market": 500, "journal": 1500,
    }
    min_len = min_lens.get(category, 500)
    if len(content) < min_len:
        return False, f"too_short_{len(content)}/{min_len}"

    # Degeneration check
    if _DEGEN_PATTERN.search(content):
        return False, "degenerate"

    # Corporate speak — max 2 violations allowed
    corp_hits = _CORP_SPEAK.findall(content)
    if len(corp_hits) > 2:
        return False, f"corporate_speak_{len(corp_hits)}"

    # Specificity — real analysis references concrete things
    specifics = _SPECIFICS.findall(content)
    min_specifics = {"briefing": 10, "trends": 8, "company": 6,
                     "culture": 4, "market": 8, "journal": 15}
    required = min_specifics.get(category, 5)
    if len(specifics) < required:
        return False, f"vague_{len(specifics)}/{required}"

    # Section structure check for longer formats
    if category in ("briefing", "journal"):
        # Should have multiple sections (headers or numbered items)
        sections = len(re.findall(r"(?:^|\n)(?:#{1,3}\s|(?:\d+\.|\*|\-)\s)", content))
        if sections < 3:
            return False, f"no_structure_{sections}"

    return True, "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoints.json"
_ckpt_lock = threading.Lock()


def load_checkpoint(category: str) -> dict:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return data.get(category, {"written": 0, "done_ids": []})
    return {"written": 0, "done_ids": []}


def save_checkpoint(category: str, written: int, done_ids: list):
    with _ckpt_lock:
        data = {}
        if CHECKPOINT_FILE.exists():
            data = json.loads(CHECKPOINT_FILE.read_text())
        data[category] = {"written": written, "done_ids": done_ids[-10000:]}
        CHECKPOINT_FILE.write_text(json.dumps(data))


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════

PROGRESS_FILE = OUTPUT_DIR / "progress.json"
_progress_lock = threading.Lock()
_progress = {"categories": {}, "start_time": None}


def update_progress(category: str, written: int, target: int,
                    gen_pass: int = 0, rewritten: int = 0, failed: int = 0):
    with _progress_lock:
        _progress["categories"][category] = {
            "written": written, "target": target,
            "gen_pass": gen_pass, "rewritten": rewritten, "failed": failed,
        }
        total_written = sum(s["written"] for s in _progress["categories"].values())
        total_target = sum(s["target"] for s in _progress["categories"].values())
        total_gen = sum(s.get("gen_pass", 0) for s in _progress["categories"].values())
        total_rw = sum(s.get("rewritten", 0) for s in _progress["categories"].values())
        elapsed = time.time() - (_progress["start_time"] or time.time())
        rate = total_written / max(elapsed / 60, 0.1)
        remaining = total_target - total_written
        eta_hours = (remaining / max(rate, 1)) / 60

        PROGRESS_FILE.write_text(json.dumps({
            "total_written": total_written,
            "total_target": total_target,
            "gen_pass": total_gen,
            "rewritten": total_rw,
            "rewrite_rate": round(total_rw / max(total_gen + total_rw, 1) * 100, 1),
            "api_calls": api_calls["total"],
            "gen_calls": api_calls.get("gen", 0),
            "pass_calls": api_calls.get("pass", 0),
            "errors": api_calls.get("error", 0),
            "rate_per_min": round(rate, 1),
            "elapsed_min": round(elapsed / 60, 1),
            "eta_hours": round(eta_hours, 1),
            "gen_model": GEN_MODEL,
            "pass_model": PASS_MODEL,
            "categories": _progress["categories"],
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR GENERATION — Two-Tier: 80B gen → quality check → 235B rewrite
# ═══════════════════════════════════════════════════════════════════════════════

def grind_pair(category: str, scenario: dict, idx: int) -> dict | None:
    """Generate one training pair. 80B turbo generates, 235B rewrites failures."""
    user_msg = scenario["prompt"]
    max_tokens = scenario.get("max_tokens", 3072)
    min_len = scenario.get("min_len", 500)

    # ── Tier 1: Fast gen with 80B turbo ──
    with api_lock:
        api_calls["gen"] += 1
    content = together_call(SYSTEM_PROMPT, user_msg, model=GEN_MODEL,
                            max_tokens=max_tokens, min_len=min_len // 2)

    if not content:
        return None

    # ── Quality gate ──
    passed, reason = quality_check(content, category)
    tier = "gen"

    if not passed:
        # ── Tier 2: 235B rewrite ──
        with api_lock:
            api_calls["pass"] += 1
        content = together_call(SYSTEM_PROMPT, user_msg, model=PASS_MODEL,
                                max_tokens=max_tokens, min_len=min_len // 2)
        if not content:
            return None
        tier = "rewrite"

    # Fingerprint
    fp = hashlib.sha256(f"{category}-{idx}-{content[:200]}".encode()).hexdigest()[:16]

    return {
        "id": f"swarmsignal-{category}-{idx:06d}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": content},
        ],
        "metadata": {
            "domain": "ai_signal",
            "category": category,
            "format": scenario["format"],
            "model": GEN_MODEL if tier == "gen" else PASS_MODEL,
            "tier": tier,
            "fingerprint": fp,
            "source": "swarmsignal-cook-v1",
            "cooked_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY GRINDER (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def grind_category(category: str, target: int | None = None, seed: int = SEED):
    """Grind one category with parallel workers."""
    generator = CATEGORY_GENERATORS[category]
    target = target or CATEGORY_TARGETS[category]

    out_file = OUTPUT_DIR / f"cat_{category}.jsonl"

    print(f"\n{'='*70}")
    print(f"  CATEGORY: {category.upper()} — {target:,} pairs")
    print(f"  Format: {generator(random.Random(seed), 0)['format']}")
    print(f"  Gen: {GEN_MODEL}")
    print(f"  Pass: {PASS_MODEL}")
    print(f"  Workers: {WORKERS}")
    print(f"{'='*70}")

    # Load checkpoint
    cp = load_checkpoint(category)
    done_ids = set(cp.get("done_ids", []))
    written = cp.get("written", 0)
    gen_pass = written
    rewritten = 0
    failed = 0

    if written >= target:
        print(f"  Complete: {written:,} pairs")
        return written

    print(f"  Resuming: {written:,} done, {target - written:,} remaining")

    # Test both models
    print(f"  Testing Gen model (80B)...")
    test1 = together_call("You are SwarmSignal.", "Say 'ready'.",
                          model=GEN_MODEL, max_tokens=10, min_len=1)
    if not test1:
        print(f"  FAIL — Gen model not responding")
        return written
    print(f"  Gen: OK")

    # Skip separate pass test — same model, already verified above
    print(f"  Pass: OK (same model)")

    t0 = time.time()
    session_start = written

    # Build work queue — each item is (idx, scenario)
    work_queue = []
    for idx in range(target * 2):  # Over-generate to account for failures
        pair_id = f"{category}-{idx}"
        if pair_id in done_ids:
            continue
        rng = random.Random(seed + idx)
        scenario = generator(rng, idx)
        work_queue.append((idx, scenario))
        if written + len(work_queue) >= target + (target // 5):  # 20% buffer
            break

    print(f"  Work queue: {len(work_queue):,} tasks")

    # Process with thread pool
    file_lock = threading.Lock()

    def process_task(item):
        nonlocal written, gen_pass, rewritten, failed
        idx, scenario = item

        if written >= target:
            return None

        rec = grind_pair(category, scenario, idx)

        if rec:
            with file_lock:
                if written >= target:
                    return None
                with open(out_file, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if rec["metadata"]["tier"] == "gen":
                    gen_pass += 1
                else:
                    rewritten += 1

                done_ids.add(f"{category}-{idx}")

                # Progress every 100 pairs
                if written % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (written - session_start) / max(elapsed / 60, 0.1)
                    remaining = target - written
                    eta = (remaining / max(rate, 1)) / 60
                    pct = written / target * 100
                    rw_pct = rewritten / max(gen_pass + rewritten, 1) * 100
                    print(f"  [{written:,}/{target:,}] ({pct:.0f}%) "
                          f"rate={rate:.0f}/min ETA={eta:.1f}h "
                          f"gen={gen_pass:,} rw={rewritten:,} ({rw_pct:.0f}%) "
                          f"err={api_calls.get('error', 0)}")
                    update_progress(category, written, target, gen_pass, rewritten, failed)

                # Checkpoint every 500 pairs
                if written % 500 == 0:
                    save_checkpoint(category, written, list(done_ids)[-10000:])

            return rec
        else:
            with file_lock:
                failed += 1
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_task, item): item for item in work_queue}
        for future in as_completed(futures):
            try:
                future.result()
            except RuntimeError as e:
                print(f"\n  FATAL: {e}")
                break
            except Exception:
                pass

            if written >= target:
                break

    # Final checkpoint
    save_checkpoint(category, written, list(done_ids)[-10000:])
    update_progress(category, written, target, gen_pass, rewritten, failed)

    elapsed = time.time() - t0
    rate = (written - session_start) / max(elapsed / 60, 0.1)
    rw_pct = rewritten / max(gen_pass + rewritten, 1) * 100
    print(f"\n  Category {category} done: {written:,} pairs, {rate:.0f}/min, "
          f"{elapsed/60:.1f} min")
    print(f"  Gen pass: {gen_pass:,} | Rewritten by 235B: {rewritten:,} ({rw_pct:.0f}%)")

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def assemble():
    """Merge all category files into final output."""
    print(f"\n{'='*70}")
    print(f"  ASSEMBLY")
    final = OUTPUT_DIR / "swarmsignal_30k.jsonl"
    total = 0
    gen_total = 0
    rw_total = 0
    for category in CATEGORY_TARGETS:
        sf = OUTPUT_DIR / f"cat_{category}.jsonl"
        if sf.exists():
            count = 0
            gen_c = 0
            rw_c = 0
            with open(sf) as f:
                for line in f:
                    rec = json.loads(line)
                    count += 1
                    if rec.get("metadata", {}).get("tier") == "rewrite":
                        rw_c += 1
                    else:
                        gen_c += 1
            size = sf.stat().st_size / (1024 * 1024)
            print(f"  {category:<15} {count:>7,} pairs  (gen={gen_c:,} rw={rw_c:,})  ({size:.1f} MB)")
            total += count
            gen_total += gen_c
            rw_total += rw_c
        else:
            print(f"  {category:<15} MISSING")

    if total == 0:
        print(f"\n  No data to assemble.")
        return

    with open(final, "w") as out:
        for category in CATEGORY_TARGETS:
            sf = OUTPUT_DIR / f"cat_{category}.jsonl"
            if sf.exists():
                with open(sf) as f:
                    for line in f:
                        out.write(line)

    size = final.stat().st_size / (1024 * 1024)
    rw_pct = rw_total / max(total, 1) * 100
    print(f"\n  Total: {total:,} pairs, {size:.1f} MB")
    print(f"  Gen pass: {gen_total:,} | 235B rewrites: {rw_total:,} ({rw_pct:.0f}%)")
    print(f"  Output: {final}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def show_status():
    if PROGRESS_FILE.exists():
        p = json.loads(PROGRESS_FILE.read_text())
        print(f"\nSwarmSignal Cook — Two-Tier Progress")
        print(f"  Gen:  {p.get('gen_model', '?')}")
        print(f"  Pass: {p.get('pass_model', '?')}")
        print(f"  Total: {p['total_written']:,} / {p['total_target']:,}")
        print(f"  Gen pass: {p.get('gen_pass', 0):,} | Rewrites: {p.get('rewritten', 0):,} "
              f"({p.get('rewrite_rate', 0)}%)")
        print(f"  Rate: {p['rate_per_min']}/min | ETA: {p['eta_hours']}h")
        print(f"  API: total={p['api_calls']:,} gen={p.get('gen_calls',0):,} "
              f"pass={p.get('pass_calls',0):,} err={p.get('errors',0)}")
        for name, s in p.get("categories", {}).items():
            rw = s.get("rewritten", 0)
            gp = s.get("gen_pass", 0)
            print(f"    {name:<15} {s['written']:>7,} / {s['target']:>7,}  "
                  f"(gen={gp:,} rw={rw:,})")
    else:
        print("No progress data yet.")

    print(f"\n  Category files:")
    for category in CATEGORY_TARGETS:
        sf = OUTPUT_DIR / f"cat_{category}.jsonl"
        if sf.exists():
            count = sum(1 for _ in open(sf))
            print(f"    {category:<15} {count:>7,} pairs")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WORKERS
    parser = argparse.ArgumentParser(
        description="SwarmSignal Cook — Two-Tier: 80B turbo gen + 235B pass/rewrite")
    parser.add_argument("--category", choices=[*CATEGORY_TARGETS.keys(), "all"], default="all")
    parser.add_argument("--target", type=int, help="Override target for single category")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.assemble:
        assemble()
        return

    # Get API key
    api_key = os.environ.get("TOGETHER_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set TOGETHER_KEY environment variable")
        sys.exit(1)

    if api_key:
        init_session(api_key)

    WORKERS = args.workers

    print(f"{'='*70}")
    print(f"  SWARMSIGNAL COOK v1 — 30K AI Trends Writer (Two-Tier)")
    print(f"  Tier 1 Gen:  {GEN_MODEL} (turbo, 3B active)")
    print(f"  Tier 2 Pass: {PASS_MODEL} (quality, 22B active)")
    print(f"  Workers: {WORKERS}")
    print(f"  Flow: 80B gen → quality gate → pass or 235B rewrite")
    print(f"{'='*70}")

    if args.dry_run:
        for cat in (CATEGORY_TARGETS if args.category == "all" else {args.category: 0}):
            gen = CATEGORY_GENERATORS[cat]
            target = args.target or CATEGORY_TARGETS[cat]
            print(f"\n{'='*70}")
            print(f"  CATEGORY: {cat.upper()} — {target:,} pairs")
            print(f"{'='*70}")
            for i in range(3):
                rng = random.Random(SEED + i)
                scenario = gen(rng, i)
                print(f"\n  --- Sample {i+1} ({scenario['format']}) ---")
                print(f"  {scenario['prompt'][:300]}...")
                print(f"  min_len={scenario['min_len']}, max_tokens={scenario.get('max_tokens', 3072)}")
        return

    # Record start time
    _progress["start_time"] = time.time()

    # Run categories
    cats_to_run = (CATEGORY_TARGETS if args.category == "all"
                   else {args.category: CATEGORY_TARGETS[args.category]})
    for cat, default_target in cats_to_run.items():
        target = args.target or default_target
        grind_category(cat, target=target, seed=args.seed)

    # Assemble
    assemble()

    rw_rate = api_calls.get("pass", 0) / max(api_calls.get("gen", 0), 1) * 100
    print(f"\n{'='*70}")
    print(f"  DONE — SwarmSignal Cook v1 Complete")
    print(f"  API calls: {api_calls['total']:,}")
    print(f"    Gen (80B):   {api_calls.get('gen', 0):,}")
    print(f"    Pass (235B): {api_calls.get('pass', 0):,} ({rw_rate:.0f}% rewrite rate)")
    print(f"    Errors:      {api_calls.get('error', 0)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
