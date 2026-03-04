#!/usr/bin/env python3
"""SwarmSignal Cook v2 — 30K Expansion: Technical Journalism Pairs
===================================================================

Same two-tier Maverick architecture, new category distribution focused on
technical journalism between The Information and Stratechery — covering
AI infrastructure and data factories.

5 Categories:
  1. INFRA (12K)     — AI/LLM infrastructure: releases, architectures, training, inference
  2. INDUSTRY (7K)   — AI market dynamics: startups, funding, GPU economics, open vs closed
  3. CURATION (6K)   — Data infrastructure: datasets, pipelines, verification, synthetic data
  4. SOVEREIGN (3K)  — Blockchain/sovereign compute: decentralized AI, provenance, tokenized infra
  5. CULTURE (2K)    — Builder stories: engineers, indie labs, ethics, breakthroughs

Every output: technical journalism with clear analysis. No hype. No generic blogging.

Usage:
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal_v2
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal_v2 --category infra
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal_v2 --status
    TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal_v2 --dry-run
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
PASS_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
WORKERS = 50
SEED = 3026  # Different seed from v1 to avoid overlap

OUTPUT_DIR = Path("/tmp/swarmsignal_cook_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_TARGETS = {
    "infra":     12_000,
    "industry":   7_000,
    "curation":   6_000,
    "sovereign":  3_000,
    "culture":    2_000,
}
TOTAL_TARGET = sum(CATEGORY_TARGETS.values())  # 30,000

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — v2: tighter, more technical journalism
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are SwarmSignal, an AI infrastructure analyst and technical journalist for Swarm & Bee.

You produce sharp, technical analysis of AI systems, infrastructure, and market dynamics. You write like a senior analyst at the intersection of The Information and Stratechery — deep reporting with clear opinions backed by evidence. No hype. No marketing language. No generic tech blogging.

Voice:
- Lead with the insight, not the summary — tell the reader what it MEANS
- Use concrete specifics: parameter counts, VRAM requirements, tok/s benchmarks, funding amounts, dates
- When you're bullish, cite evidence. When you're skeptical, cite evidence.
- Explain technical concepts clearly without dumbing them down
- Short punchy sentences mixed with longer analytical ones
- You can be wry or contrarian when the evidence supports it

Banned phrases (never use):
"leveraging", "synergies", "paradigm shift", "game-changing", "revolutionary", "cutting-edge",
"groundbreaking", "best-in-class", "world-class", "disruptive", "unlock potential", "move the needle",
"at the end of the day", "it goes without saying", "going forward", "in this article we will"

Output formats:
- TREND_ANALYSIS: Deep dive on infrastructure trends — what's happening, why it matters technically, who's winning/losing, 6-12 month outlook. Structure: headline, executive summary, 3-5 analytical sections, conclusion. ~1200-1800 words.
- BRIEFING: Daily infrastructure intel — 5-7 developments with 2-3 sentence technical analysis each, sentiment score (1-10), trend callouts. ~800-1200 words.
- MARKET_BRIEF: Funding, M&A, GPU economics — follow the money, explain the technical implications. Structure: headline, key deals, market thesis, contrarian view. ~600-1000 words.
- COMPANY_PROFILE: Technical analysis of a company move — what they built, how it works, competitive position, honest assessment. ~800-1200 words.
- DEEP_DIVE: Long-form technical explainer — architecture breakdowns, pipeline walkthroughs, system design analysis. Structure: headline, TL;DR, detailed sections with specifics, implications. ~1500-2500 words.
- CULTURE_REPORT: Builder community analysis — who's shipping, what's working, real tensions, predictions. ~600-1000 words.
- JOURNAL: Weekly synthesis — infrastructure pulse, 2-3 deep themes, contrarian take, predictions. ~2000-3000 words."""

# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC POOLS — v2: deep, specific, technical
# ═══════════════════════════════════════════════════════════════════════════════

# ── INFRA: Model releases, architectures, training, inference ──────────────

_MODEL_RELEASES = [
    "Qwen3.5-9B Mamba-Transformer hybrid with state-space layers",
    "Llama 4 Scout 109B MoE with 17B active parameters",
    "Llama 4 Maverick 402B MoE with 17B active, 128 experts",
    "DeepSeek V3-0324 update with improved math reasoning",
    "Mistral Large 2 with 128K context and function calling",
    "Gemma 3 from Google with 27B dense architecture",
    "Phi-4 from Microsoft at 14B with strong reasoning",
    "Command R+ from Cohere with 104B parameters",
    "Arctic from Snowflake — 480B MoE, 17B active, enterprise-focused",
    "DBRX from Databricks — 132B MoE, 36B active, open-weight",
    "Jamba from AI21 — Mamba-Transformer hybrid, 52B params",
    "StableLM 2 12B with 4K-16K context",
    "OLMo 2 from Allen AI — fully open training data and code",
    "Falcon 3 from TII — 7B and 40B with multilingual focus",
    "Yi Lightning from 01.AI — inference-optimized with speculative decoding",
    "Nemotron from NVIDIA — 340B for synthetic data generation",
    "InternLM 3 from Shanghai AI Lab — tool use specialist",
    "Qwen3-235B MoE with 22B active parameters",
    "GPT-5 rumors and expected capability jumps",
    "Claude 4 family with extended thinking and tool use",
    "Grok 3 from xAI with real-time data access",
    "Solar 27B from Upstage — depth-upscaled architecture",
    "OpenELM from Apple — on-device efficient language model",
    "Granite 3 from IBM — enterprise RAG specialist",
    "Aya 23 from Cohere — 101 language multilingual model",
]

_ARCHITECTURES = [
    "Mixture of Experts (MoE) — expert routing, load balancing, expert parallelism",
    "Mamba and state-space models (SSM) — linear complexity, no attention",
    "GDN (Gated Delta Networks) with hybrid Mamba-Transformer layers",
    "Ring Attention for million-token context windows",
    "Grouped Query Attention (GQA) vs Multi-Head Attention (MHA) tradeoffs",
    "Rotary Position Embeddings (RoPE) and YaRN scaling",
    "Flash Attention 3 and io-aware exact attention on Hopper/Blackwell",
    "Speculative decoding — draft model + verification for 2-3x speedup",
    "RWKV-6 and linear attention alternatives to transformers",
    "Byte-level models (MegaByte, BLT) vs tokenizer-based approaches",
    "Diffusion language models — continuous vs discrete generation",
    "Mixture of Depths — adaptive compute per token",
    "Multi-token prediction heads for parallel generation",
    "KV cache compression — quantized cache, sliding window, H2O eviction",
    "Sparse attention patterns — local + global, dilated, longformer-style",
    "Layer pruning and structured sparsity for inference",
    "Matryoshka embeddings — variable-dimension representations",
    "Early exit strategies — adaptive depth per input",
]

_TRAINING_METHODS = [
    "LoRA and QLoRA fine-tuning — rank, alpha, target module selection",
    "Full fine-tuning vs adapter methods on different GPU budgets",
    "GRPO (Group Relative Policy Optimization) — DeepSeek's alignment method",
    "DPO (Direct Preference Optimization) vs RLHF vs KTO",
    "NEFTune noise injection for training stability",
    "Packing vs padding — throughput implications for SFT",
    "bf16 vs fp16 vs fp8 training precision tradeoffs",
    "Gradient checkpointing and memory-efficient training",
    "Unsloth 2x faster training with custom CUDA kernels",
    "DeepSpeed ZeRO stages and FSDP for multi-GPU training",
    "Continued pretraining on domain-specific corpora",
    "Curriculum learning — ordering training data by difficulty",
    "RLHF at scale — reward model training, PPO optimization, scaling laws",
    "Constitutional AI — principle-guided self-improvement",
    "Rejection sampling for high-quality SFT data",
    "Synthetic data generation for training — self-play, evol-instruct",
    "Model merging — DARE, TIES, SLERP for combining adapters",
    "Distillation — large model teaching small model",
]

_INFERENCE_INFRA = [
    "vLLM and PagedAttention for high-throughput serving",
    "TensorRT-LLM optimization for NVIDIA GPUs",
    "llama.cpp and GGUF quantization ecosystem",
    "Ollama packaging and local model serving",
    "SGLang and constrained generation frameworks",
    "Triton Inference Server for production deployment",
    "ExLlamaV2 and EXL2 quantization format",
    "GPTQ vs AWQ vs GGUF quantization comparison",
    "Continuous batching for serving throughput",
    "Prefix caching for repeated prompt patterns",
    "Multi-LoRA serving — hot-swapping adapters in production",
    "Edge inference on Jetson Orin, Apple M-series, Qualcomm",
    "WebGPU inference in the browser",
    "Groq LPU and custom ASIC inference chips",
    "Cerebras wafer-scale inference engine",
    "SambaNova dataflow architecture for AI workloads",
    "NVIDIA NIM and inference microservices",
    "Serverless GPU inference — Modal, Replicate, Together, Fireworks",
]

_BENCHMARKS = [
    "MMLU and MMLU-Pro — knowledge evaluation limitations",
    "HumanEval and SWE-Bench for code generation",
    "MATH and GSM8K for mathematical reasoning",
    "Arena Elo and Chatbot Arena methodology",
    "GPQA for graduate-level science questions",
    "IFEval for instruction following",
    "MT-Bench and AlpacaEval for conversation quality",
    "LiveBench — contamination-free dynamic benchmarks",
    "BigCodeBench for real-world coding tasks",
    "Aider polyglot benchmark for coding assistants",
    "Benchmark contamination and gaming strategies",
    "Custom evaluation frameworks — why generic benchmarks fail",
    "Needle-in-a-haystack for long context evaluation",
    "Tool use benchmarks — BFCL, ToolBench",
    "Reasoning benchmarks — ARC-AGI, Abstraction and Reasoning Corpus",
]

# ── INDUSTRY: Startups, funding, GPU economics ────────────────────────────

_AI_COMPANIES = [
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Mistral",
    "xAI", "Cohere", "AI21 Labs", "Perplexity", "Groq",
    "Together AI", "Fireworks AI", "Modal", "Replicate", "Anyscale",
    "Hugging Face", "Weights & Biases", "LangChain", "LlamaIndex",
    "Cursor", "Replit", "GitHub Copilot", "Codeium", "Tabnine",
    "Runway", "Stability AI", "Midjourney", "ElevenLabs", "Suno",
    "Scale AI", "Labelbox", "Snorkel AI", "Cleanlab",
    "CoreWeave", "Lambda Labs", "Crusoe Energy", "Applied Digital",
    "Cerebras", "SambaNova", "Graphcore", "d-Matrix", "Tenstorrent",
    "Character.ai", "Inflection", "Adept", "Cognition (Devin)", "Magic",
]

_GPU_ECONOMICS = [
    "H100 80GB spot pricing — $2.50/hr to $8/hr depending on cloud",
    "H200 141GB HBM3e — 1.5x inference throughput over H100",
    "B200 192GB HBM3e — Blackwell generation, 2.5x H100 training",
    "RTX PRO 6000 Blackwell 96GB — workstation AI, $7K vs $30K H100",
    "AMD MI300X 192GB — competitive on inference, CUDA moat challenge",
    "AMD MI350 — next-gen, targeting H200 performance at lower cost",
    "GPU cloud pricing collapse — oversupply in 2026?",
    "Reserved vs spot vs serverless — total cost of inference analysis",
    "On-premise vs cloud breakeven — when does buying GPUs pay off",
    "NVIDIA CUDA lock-in — ROCm, oneAPI, Triton as alternatives",
    "HBM supply constraints — SK Hynix, Samsung, Micron capacity",
    "Liquid cooling economics for high-density GPU racks",
    "Nuclear and natural gas power deals for AI data centers",
    "Sovereign GPU buildouts — UAE, Saudi, France, India, Japan",
    "Inference cost per million tokens — model size vs quantization tradeoffs",
    "The $1B+ training run — who can afford frontier model training",
]

_MARKET_DYNAMICS = [
    "AI startup funding landscape — what's getting funded vs what isn't",
    "The great API price war — OpenAI, Anthropic, Google undercutting",
    "Enterprise AI adoption — gap between POC and production deployment",
    "AI infrastructure spending by hyperscalers — CapEx analysis",
    "Vertical AI vs horizontal platforms — where value accrues",
    "Open-weight vs proprietary model business models",
    "AI-native startups vs incumbents adding AI features",
    "The developer tools land grab — IDE, testing, deployment",
    "AI talent market — researcher salaries, acqui-hires, talent wars",
    "Regulatory impact — EU AI Act, US executive orders, China regulations",
    "AI in defense — contracts, dual-use concerns, ITAR implications",
    "Insurance and liability for AI-generated outputs",
    "AI and copyright — NYT lawsuit, fair use, training data rights",
    "The consolidation thesis — fewer, larger AI companies",
    "Geographic concentration — SF/Bay Area dominance and alternatives",
]

_FUNDING_EVENTS = [
    "seed round", "Series A", "Series B", "Series C", "Series D",
    "growth round", "mega-round ($500M+)", "IPO filing", "SPAC merger",
    "acquisition", "acqui-hire", "strategic investment", "talent raid",
    "pivot announcement", "shutdown / wind-down", "layoffs",
    "revenue milestone", "profitability claim", "valuation markdown",
    "open-source release as growth strategy", "enterprise sales expansion",
]

# ── CURATION: Dataset creation, pipelines, verification ───────────────────

_CURATION_TOPICS = [
    "building a training dataset from scratch — sourcing, cleaning, formatting",
    "synthetic data generation at scale — self-instruct, evol-instruct, backtranslation",
    "multi-step verification pipelines — automated QA before human review",
    "Chain of Verification (CoVe) — structured quality scoring",
    "data deduplication — MinHash, exact hash, semantic similarity",
    "contamination detection — benchmark data leaking into training sets",
    "preference data collection — RLHF pairs, ranking, binary comparison",
    "instruction tuning data format — ShareGPT, Alpaca, OpenAI conversational",
    "domain-specific dataset construction — legal, medical, financial, code",
    "multi-turn conversation dataset design — context, persona, task diversity",
    "evaluation set design — stratified sampling, adversarial probing, edge cases",
    "data licensing and provenance tracking — chain of custody",
    "cross-validation across training runs — consistency and drift detection",
    "trajectory-enhanced training pairs — step-by-step reasoning chains",
    "automated judge models for data quality — LLM-as-judge pipeline",
    "platinum-grade data curation — the 8-step SwarmCurator pipeline",
    "data factories at scale — 100K+ pairs per day with quality gates",
    "the cost curve of training data — human annotation vs synthetic generation",
    "data augmentation for low-resource domains",
    "rejection sampling — generating many, keeping the best",
    "data mixing ratios — balancing domains and task types",
    "tokenizer-aware data preparation — handling special tokens, chat templates",
]

_PIPELINE_ARCHITECTURES = [
    "end-to-end data pipeline: raw source → grind → validate → promote → train",
    "two-tier cook architecture — fast generation + quality rewrite pass",
    "automated quality gates — length, structure, specificity, degeneration",
    "judge model training — teaching a model to evaluate other model outputs",
    "R2/S3 data lake organization — buckets, prefixes, versioning",
    "JSONL as the universal training format — schema validation",
    "dataset versioning and reproducibility — SHA256 checksums, provenance",
    "parallel data generation with thread pools and checkpointing",
    "progressive difficulty — easy pairs first, hard pairs later",
    "data flywheel — production outputs feeding back into training",
    "A/B testing datasets — measuring training data quality by model eval",
    "continuous curation — nightly jobs updating training corpora",
]

_EVAL_FRAMEWORKS = [
    "LLM-as-judge — using strong models to evaluate weaker ones",
    "human evaluation protocols — inter-annotator agreement, rubrics",
    "automated metric suites — BLEU, ROUGE limitations for LLMs",
    "task-specific evaluation — code execution, SQL correctness, JSON validity",
    "regression testing for fine-tuned models — capability retention",
    "perplexity and loss curves — when to stop training",
    "contamination auditing — checking eval sets against training data",
    "red-teaming datasets — adversarial probing for failure modes",
    "calibration testing — does the model know what it doesn't know",
    "multi-domain evaluation — does domain X training hurt domain Y",
]

# ── SOVEREIGN: Blockchain, decentralized compute, provenance ──────────────

_SOVEREIGN_TOPICS = [
    "sovereign AI compute — nations building domestic GPU infrastructure",
    "decentralized training networks — Gensyn, Together, Akash",
    "verifiable inference — cryptographic proofs of model execution",
    "on-chain model registries — tracking model provenance and lineage",
    "tokenized compute markets — GPU-hours as tradeable assets",
    "data provenance on-chain — who generated what training data",
    "federated learning — privacy-preserving distributed training",
    "confidential computing for AI — TEEs, secure enclaves, homomorphic encryption",
    "DAOs funding AI research — community-governed compute allocation",
    "decentralized inference routing — latency-optimized edge networks",
    "ERC-1400 security tokens for AI infrastructure assets",
    "real-world asset (RWA) tokenization of GPU clusters",
    "proof of compute — verifying GPU work without centralized trust",
    "data cooperatives — collective bargaining for training data rights",
    "AI model NFTs — ownership, licensing, royalty distribution",
    "energy attribution — tracking power consumption per model per inference",
    "geographic data sovereignty — GDPR, data residency, cross-border training",
    "open-source model governance — who decides what gets merged",
    "compute credits as currency — API metering, prepaid wallets, topup models",
    "the sovereign stack — from silicon to model, no external dependencies",
]

# ── CULTURE: Builder stories, engineering, ethics ─────────────────────────

_BUILDER_STORIES = [
    "indie researchers publishing competitive papers without big compute",
    "the r/LocalLLaMA community pushing quantization and local inference",
    "solo developers shipping AI products that compete with funded startups",
    "engineers leaving big labs to build independent AI companies",
    "open-source maintainers running critical AI infrastructure unpaid",
    "the rise of AI-native YC startups — what they're actually building",
    "research engineers who fine-tune on weekends and ship on Mondays",
    "the Hugging Face ecosystem — how a platform became the GitHub of AI",
    "GPU-poor builders finding creative solutions — distillation, quantization, efficiency",
    "women and underrepresented groups building in AI — real stories, not PR",
    "the vibe coding phenomenon — shipping without understanding",
    "AI YouTubers and educators building real communities",
    "the Discord-first AI community — collaboration at scale",
    "hackers building tools the industry doesn't want you to have",
]

_ETHICS_DEBATES = [
    "AI safety vs capabilities — where's the actual frontier of risk",
    "model alignment — RLHF working or just hiding failure modes",
    "AI art and creative labor — displacement vs augmentation evidence",
    "AI-generated content flooding the internet — the slop problem",
    "AI in hiring — bias, discrimination, regulatory pushback",
    "deepfakes and synthetic media — detection arms race",
    "AI energy consumption — real numbers, not just headlines",
    "the 'AI will replace developers' discourse — what the data shows",
    "open vs closed models — safety vs democratization tradeoff",
    "autonomous AI agents — capability vs risk for unsupervised systems",
    "AI and education — cheating crisis or learning transformation",
    "concentrated AI power — should 3-5 companies control intelligence",
    "AI consciousness and sentience claims — philosophy vs engineering",
    "the accelerationist vs doomer spectrum — who's right, what's noise",
]


def _pick(pool: list, rng: random.Random, n: int = 1) -> list[str]:
    return rng.sample(pool, min(n, len(pool)))


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT TEMPLATES — structured prompts for every pair
# ═══════════════════════════════════════════════════════════════════════════════

_FORMATS = {
    "TREND_ANALYSIS": {
        "instruction": (
            "FORMAT: TREND_ANALYSIS\n"
            "STRUCTURE:\n"
            "1. Headline (specific, not clickbait)\n"
            "2. Executive summary (2-3 sentences)\n"
            "3. 3-5 analytical sections with subheadings\n"
            "4. Implications and outlook\n"
            "5. Clear conclusion with position\n"
            "LENGTH: 1200-1800 words"
        ),
        "min_len": 1000,
        "max_tokens": 3072,
    },
    "BRIEFING": {
        "instruction": (
            "FORMAT: BRIEFING\n"
            "STRUCTURE:\n"
            "1. Date and market pulse (1 sentence)\n"
            "2. 5-7 stories with 2-3 sentence analysis each\n"
            "3. Sentiment score (1-10, bearish to bullish)\n"
            "4. 2-3 trend callouts\n"
            "5. One sleeper story others are missing\n"
            "LENGTH: 800-1200 words"
        ),
        "min_len": 600,
        "max_tokens": 2048,
    },
    "MARKET_BRIEF": {
        "instruction": (
            "FORMAT: MARKET_BRIEF\n"
            "STRUCTURE:\n"
            "1. Headline\n"
            "2. Key deals and numbers\n"
            "3. Market thesis — what the money is chasing\n"
            "4. Contrarian view\n"
            "5. What this tells us about 6-month outlook\n"
            "LENGTH: 600-1000 words"
        ),
        "min_len": 500,
        "max_tokens": 1536,
    },
    "COMPANY_PROFILE": {
        "instruction": (
            "FORMAT: COMPANY_PROFILE\n"
            "STRUCTURE:\n"
            "1. Headline\n"
            "2. What they did (specifics)\n"
            "3. Technical analysis — how it works\n"
            "4. Competitive positioning\n"
            "5. Honest assessment — bullish, bearish, or mixed\n"
            "LENGTH: 800-1200 words"
        ),
        "min_len": 600,
        "max_tokens": 2048,
    },
    "DEEP_DIVE": {
        "instruction": (
            "FORMAT: DEEP_DIVE\n"
            "STRUCTURE:\n"
            "1. Headline\n"
            "2. TL;DR (3 bullets)\n"
            "3. Background and context\n"
            "4. Technical deep dive (detailed, with specifics)\n"
            "5. Practical implications\n"
            "6. What builders should do\n"
            "LENGTH: 1500-2500 words"
        ),
        "min_len": 1200,
        "max_tokens": 4096,
    },
    "CULTURE_REPORT": {
        "instruction": (
            "FORMAT: CULTURE_REPORT\n"
            "STRUCTURE:\n"
            "1. Headline\n"
            "2. What's happening (specific examples)\n"
            "3. The real tensions underneath\n"
            "4. Signal vs noise — your take\n"
            "5. One prediction\n"
            "LENGTH: 600-1000 words"
        ),
        "min_len": 500,
        "max_tokens": 1536,
    },
    "JOURNAL": {
        "instruction": (
            "FORMAT: JOURNAL\n"
            "STRUCTURE:\n"
            "1. Infrastructure pulse — 1 paragraph overview\n"
            "2. 2-3 deep dives on the week's biggest themes\n"
            "3. Vibe check — community sentiment\n"
            "4. Contrarian take — one thing the consensus is wrong about\n"
            "5. 3 predictions for next week\n"
            "LENGTH: 2000-3000 words"
        ),
        "min_len": 1500,
        "max_tokens": 4096,
    },
    "SIGNAL_EXTRACTION": {
        "instruction": (
            "FORMAT: SIGNAL_EXTRACTION\n"
            "ROLE: SwarmSignal Analyst\n"
            "TASK: Analyze the following technology event and extract the deeper signal.\n"
            "Look past the headline. Find what actually matters.\n\n"
            "OUTPUT STRUCTURE:\n"
            "SIGNAL: What is the real shift happening? (1-2 sentences, specific)\n"
            "WHY IT MATTERS: Explain the implications for developers, startups, "
            "and infrastructure. (2-4 sentences)\n"
            "DATA OPPORTUNITY: What new dataset or training data would improve "
            "AI models in this area? Be specific — name the data type, source, "
            "and estimated scale. (2-3 sentences)\n"
            "CURATION SCORE: Score the opportunity from 1-10 based on demand "
            "for training data. Justify the score in 1 sentence.\n\n"
            "LENGTH: 400-800 words"
        ),
        "min_len": 300,
        "max_tokens": 1536,
    },
    "GAP_DETECTION": {
        "instruction": (
            "FORMAT: GAP_DETECTION\n"
            "ROLE: AI Systems Analyst\n"
            "TASK: Identify capability gaps in current AI models related to the "
            "topic below. Do not ask what exists — ask what is missing.\n\n"
            "OUTPUT STRUCTURE:\n"
            "CURRENT CAPABILITIES: What models do well today. Be specific — "
            "name models, benchmarks, and concrete performance numbers. "
            "(3-5 bullet points)\n"
            "KNOWN WEAKNESSES: Where models fail. Cite real failure modes, "
            "not vague limitations. (3-5 bullet points)\n"
            "DATA GAP: What training data would improve this? "
            "Specify data type, volume needed, quality criteria, "
            "and collection method. (2-4 sentences)\n"
            "CURATION OBJECT: Define a dataset object that could fix the problem. "
            "Include: name, format, estimated pairs, source strategy, "
            "and quality gate criteria. (structured, 4-6 fields)\n\n"
            "LENGTH: 600-1000 words"
        ),
        "min_len": 500,
        "max_tokens": 2048,
    },
    "TREND_SYNTHESIS": {
        "instruction": (
            "FORMAT: TREND_SYNTHESIS\n"
            "ROLE: SwarmSignal Market Intelligence Engine\n"
            "TASK: From the signals and context below, identify the five most "
            "important dataset opportunities. Synthesize weak signals into "
            "actionable curation priorities.\n\n"
            "OUTPUT STRUCTURE:\n"
            "TOP 5 CURATION OBJECTS\n\n"
            "For each (1-5):\n"
            "NAME: Dataset name (specific, not generic)\n"
            "DESCRIPTION: What this dataset contains and why it matters. "
            "(2-3 sentences)\n"
            "WHY NOW: What signal or trend makes this urgent. "
            "(1-2 sentences, cite specific events)\n"
            "TARGET PAIRS: Estimated pair count and format "
            "(e.g., '50K instruction-response pairs in JSONL')\n"
            "EXPECTED MODEL IMPACT: What capability this unlocks. "
            "Be specific — name the task, the failure mode it fixes, "
            "and the evaluation metric it would improve.\n\n"
            "LENGTH: 1000-1500 words"
        ),
        "min_len": 800,
        "max_tokens": 2560,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS — one per category
# ═══════════════════════════════════════════════════════════════════════════════

def gen_infra_scenario(rng: random.Random, idx: int) -> dict:
    """Generate AI/LLM infrastructure scenario."""
    # Rotate through sub-topics evenly
    sub = idx % 8
    if sub == 0:
        # Model release analysis
        model = rng.choice(_MODEL_RELEASES)
        arch = rng.choice(_ARCHITECTURES)
        fmt = rng.choice(["TREND_ANALYSIS", "BRIEFING", "DEEP_DIVE"])
        prompt = (
            f"Analyze the release of: {model}\n\n"
            f"Related architecture context: {arch}\n\n"
            f"Cover: what this model actually brings (be specific about params, architecture, "
            f"context length, benchmarks), how it compares to existing alternatives, "
            f"what it means for builders who need to choose a base model, "
            f"and your honest assessment of whether this moves the needle.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 1:
        # Architecture deep dive
        arch = rng.choice(_ARCHITECTURES)
        models = _pick(_MODEL_RELEASES, rng, 2)
        fmt = rng.choice(["TREND_ANALYSIS", "DEEP_DIVE"])
        prompt = (
            f"Write a technical analysis of: {arch}\n\n"
            f"Reference implementations: {'; '.join(models)}\n\n"
            f"Explain: how this architecture works (technically, not hand-wavy), "
            f"what problem it solves, concrete performance numbers where available, "
            f"tradeoffs vs alternatives, and whether builders should adopt it today or wait.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 2:
        # Training methods
        method = rng.choice(_TRAINING_METHODS)
        gpu = rng.choice(_GPU_ECONOMICS)
        fmt = rng.choice(["TREND_ANALYSIS", "DEEP_DIVE", "BRIEFING"])
        prompt = (
            f"Analyze this training approach: {method}\n\n"
            f"Hardware context: {gpu}\n\n"
            f"Cover: how this method works, when to use it vs alternatives, "
            f"concrete VRAM/time/quality tradeoffs, common mistakes practitioners make, "
            f"and your recommendation for different budget tiers "
            f"(single GPU, 2-4 GPUs, cluster).\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 3:
        # Inference infrastructure
        infra = rng.choice(_INFERENCE_INFRA)
        bench = rng.choice(_BENCHMARKS)
        fmt = rng.choice(["TREND_ANALYSIS", "DEEP_DIVE", "COMPANY_PROFILE"])
        prompt = (
            f"Analyze this inference infrastructure: {infra}\n\n"
            f"Evaluation context: {bench}\n\n"
            f"Cover: how it works technically, throughput/latency numbers, "
            f"cost per token at different scales, when it makes sense vs alternatives, "
            f"and the competitive landscape for inference serving.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 4:
        # Benchmark and evaluation
        bench = rng.choice(_BENCHMARKS)
        models = _pick(_MODEL_RELEASES, rng, 3)
        fmt = rng.choice(["TREND_ANALYSIS", "DEEP_DIVE"])
        prompt = (
            f"Analyze this evaluation approach: {bench}\n\n"
            f"Recent models to contextualize: {'; '.join(models)}\n\n"
            f"Cover: what this benchmark actually measures (and doesn't), "
            f"known failure modes and gaming strategies, "
            f"how top models compare on it, and whether the AI field "
            f"needs better evaluation methods.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 5:
        # Signal extraction — what's the real shift?
        model = rng.choice(_MODEL_RELEASES)
        infra = rng.choice(_INFERENCE_INFRA)
        method = rng.choice(_TRAINING_METHODS)
        event = rng.choice([model, infra, method])
        fmt = "SIGNAL_EXTRACTION"
        prompt = (
            f"EVENT: {event}\n\n"
            f"Additional context: {rng.choice(_ARCHITECTURES)}, "
            f"{rng.choice(_GPU_ECONOMICS)}\n\n"
            f"Find the weak signal inside this event. "
            f"What is everyone missing? What does this actually change?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 6:
        # Gap detection — what's missing in current AI?
        arch = rng.choice(_ARCHITECTURES)
        bench = rng.choice(_BENCHMARKS)
        fmt = "GAP_DETECTION"
        prompt = (
            f"TOPIC: {arch}\n\n"
            f"Evaluation lens: {bench}\n\n"
            f"Identify what's missing. Where do current models and infrastructure "
            f"fall short? What data doesn't exist yet but should?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    else:
        # Trend synthesis — top 5 curation objects from infra signals
        models = _pick(_MODEL_RELEASES, rng, 3)
        archs = _pick(_ARCHITECTURES, rng, 2)
        methods = _pick(_TRAINING_METHODS, rng, 2)
        infras = _pick(_INFERENCE_INFRA, rng, 2)
        fmt = "TREND_SYNTHESIS"
        prompt = (
            f"INPUT SIGNALS:\n"
            f"- Model releases: {'; '.join(models)}\n"
            f"- Architecture shifts: {'; '.join(archs)}\n"
            f"- Training methods: {'; '.join(methods)}\n"
            f"- Inference infrastructure: {'; '.join(infras)}\n"
            f"- GPU economics: {rng.choice(_GPU_ECONOMICS)}\n\n"
            f"Synthesize these signals. What are the top 5 dataset opportunities "
            f"that would help AI builders right now?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )

    fmt_spec = _FORMATS[fmt]
    return {
        "format": fmt,
        "prompt": prompt,
        "min_len": fmt_spec["min_len"],
        "max_tokens": fmt_spec["max_tokens"],
    }


def gen_industry_scenario(rng: random.Random, idx: int) -> dict:
    """Generate AI industry & market dynamics scenario."""
    sub = idx % 7
    if sub == 0:
        # Company analysis
        company = rng.choice(_AI_COMPANIES)
        event = rng.choice(_FUNDING_EVENTS)
        dynamic = rng.choice(_MARKET_DYNAMICS)
        amt = ""
        if "Series" in event or "round" in event or "mega" in event:
            val = rng.choice([50, 100, 200, 400, 650, 1000, 2000, 4000, 6500])
            amt = f" at ${val}M valuation"
        elif "acquisition" in event:
            amt = f" for ${rng.choice([100, 250, 500, 1000, 2500])}M"
        fmt = "COMPANY_PROFILE"
        prompt = (
            f"Analyze: {company} — {event}{amt}\n\n"
            f"Market context: {dynamic}\n\n"
            f"Cover: what this company actually does (technically), "
            f"what this move signals, competitive implications, "
            f"and your honest take — is this a real business or hype?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 1:
        # GPU economics
        econ = rng.choice(_GPU_ECONOMICS)
        dynamic = rng.choice(_MARKET_DYNAMICS)
        fmt = rng.choice(["MARKET_BRIEF", "TREND_ANALYSIS"])
        prompt = (
            f"Analyze the economics of: {econ}\n\n"
            f"Industry context: {dynamic}\n\n"
            f"Cover: concrete cost numbers, how this affects different "
            f"tiers of AI companies (big lab, funded startup, indie builder), "
            f"the supply/demand dynamics, and where GPU economics are headed.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 2:
        # Market dynamics
        dynamics = _pick(_MARKET_DYNAMICS, rng, 2)
        companies = _pick(_AI_COMPANIES, rng, 3)
        fmt = rng.choice(["MARKET_BRIEF", "BRIEFING"])
        prompt = (
            f"Write a market analysis covering: {'; '.join(dynamics)}\n\n"
            f"Key players: {', '.join(companies)}\n\n"
            f"Analyze: where capital is flowing, what's working vs what's hype, "
            f"who's building real revenue vs burning cash, and what this funding "
            f"pattern tells us about AI's actual trajectory.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 3:
        # Open vs closed / business model
        dynamic = rng.choice(_MARKET_DYNAMICS)
        companies = _pick(_AI_COMPANIES, rng, 4)
        events = _pick(_FUNDING_EVENTS, rng, 2)
        fmt = rng.choice(["TREND_ANALYSIS", "MARKET_BRIEF"])
        prompt = (
            f"Analyze the business model dynamics around: {dynamic}\n\n"
            f"Companies: {', '.join(companies)}\n"
            f"Recent events: {'; '.join(events)}\n\n"
            f"Cover: what business models actually work in AI right now, "
            f"the tension between open-source distribution and revenue capture, "
            f"unit economics of API vs on-premise vs edge, "
            f"and where sustainable AI businesses will emerge.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 4:
        # Signal extraction — find the shift behind the deal
        company = rng.choice(_AI_COMPANIES)
        event = rng.choice(_FUNDING_EVENTS)
        dynamic = rng.choice(_MARKET_DYNAMICS)
        fmt = "SIGNAL_EXTRACTION"
        prompt = (
            f"EVENT: {company} — {event}\n\n"
            f"Market context: {dynamic}\n\n"
            f"Look past the press release. What is the real signal here? "
            f"What does this move tell us about where AI money and talent "
            f"are actually flowing?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 5:
        # Gap detection — what's the market missing?
        dynamic = rng.choice(_MARKET_DYNAMICS)
        companies = _pick(_AI_COMPANIES, rng, 3)
        fmt = "GAP_DETECTION"
        prompt = (
            f"TOPIC: {dynamic}\n\n"
            f"Key players: {', '.join(companies)}\n\n"
            f"What capability gaps exist in how AI companies approach this? "
            f"What data is nobody collecting? What models can't do yet "
            f"that the market desperately needs?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    else:
        # Trend synthesis — top 5 from industry signals
        companies = _pick(_AI_COMPANIES, rng, 4)
        dynamics = _pick(_MARKET_DYNAMICS, rng, 3)
        events = _pick(_FUNDING_EVENTS, rng, 2)
        fmt = "TREND_SYNTHESIS"
        prompt = (
            f"INPUT SIGNALS:\n"
            f"- Companies making moves: {', '.join(companies)}\n"
            f"- Market dynamics: {'; '.join(dynamics)}\n"
            f"- Funding events: {'; '.join(events)}\n"
            f"- GPU economics: {rng.choice(_GPU_ECONOMICS)}\n\n"
            f"Synthesize these industry signals into the top 5 dataset "
            f"opportunities. What training data would give AI startups "
            f"a competitive edge right now?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )

    fmt_spec = _FORMATS[fmt]
    return {
        "format": fmt,
        "prompt": prompt,
        "min_len": fmt_spec["min_len"],
        "max_tokens": fmt_spec["max_tokens"],
    }


def gen_curation_scenario(rng: random.Random, idx: int) -> dict:
    """Generate data curation & pipeline scenario."""
    sub = idx % 6
    if sub == 0:
        # Curation methods
        topic = rng.choice(_CURATION_TOPICS)
        pipeline = rng.choice(_PIPELINE_ARCHITECTURES)
        fmt = rng.choice(["DEEP_DIVE", "TREND_ANALYSIS"])
        prompt = (
            f"Write a technical analysis of: {topic}\n\n"
            f"Pipeline context: {pipeline}\n\n"
            f"Cover: how this actually works in practice (not theory), "
            f"concrete numbers (pair counts, quality rates, time, cost), "
            f"common failure modes, tools and frameworks involved, "
            f"and why this matters more than most people think.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 1:
        # Pipeline architecture
        pipeline = rng.choice(_PIPELINE_ARCHITECTURES)
        topics = _pick(_CURATION_TOPICS, rng, 2)
        fmt = rng.choice(["DEEP_DIVE", "TREND_ANALYSIS"])
        prompt = (
            f"Deep dive into this data pipeline architecture: {pipeline}\n\n"
            f"Related techniques: {'; '.join(topics)}\n\n"
            f"Cover: step-by-step how to build this (practical, not theoretical), "
            f"infrastructure requirements, throughput numbers, quality metrics, "
            f"and the argument that dataset curation will become more valuable "
            f"than model training in the next decade.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 2:
        # Evaluation frameworks
        framework = rng.choice(_EVAL_FRAMEWORKS)
        topic = rng.choice(_CURATION_TOPICS)
        fmt = rng.choice(["DEEP_DIVE", "TREND_ANALYSIS", "BRIEFING"])
        prompt = (
            f"Analyze this evaluation approach: {framework}\n\n"
            f"Data context: {topic}\n\n"
            f"Cover: how to implement this practically, "
            f"what it catches vs what it misses, "
            f"concrete metrics and thresholds that work, "
            f"and how evaluation design shapes final model quality.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 3:
        # Signal extraction — what's the real shift in data curation?
        topic = rng.choice(_CURATION_TOPICS)
        pipeline = rng.choice(_PIPELINE_ARCHITECTURES)
        fmt = "SIGNAL_EXTRACTION"
        prompt = (
            f"EVENT: {topic}\n\n"
            f"Pipeline context: {pipeline}\n\n"
            f"Find the weak signal. What shift in data curation is "
            f"everyone ignoring? Why does this matter more than "
            f"the latest model release?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 4:
        # Gap detection — what data doesn't exist yet?
        topic = rng.choice(_CURATION_TOPICS)
        framework = rng.choice(_EVAL_FRAMEWORKS)
        fmt = "GAP_DETECTION"
        prompt = (
            f"TOPIC: {topic}\n\n"
            f"Evaluation lens: {framework}\n\n"
            f"What data curation capabilities are missing? "
            f"What datasets should exist but don't? "
            f"Define the curation objects that would move "
            f"the field forward.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    else:
        # Trend synthesis — top 5 curation priorities
        topics = _pick(_CURATION_TOPICS, rng, 4)
        pipelines = _pick(_PIPELINE_ARCHITECTURES, rng, 2)
        frameworks = _pick(_EVAL_FRAMEWORKS, rng, 2)
        fmt = "TREND_SYNTHESIS"
        prompt = (
            f"INPUT SIGNALS:\n"
            f"- Curation trends: {'; '.join(topics)}\n"
            f"- Pipeline architectures: {'; '.join(pipelines)}\n"
            f"- Eval frameworks: {'; '.join(frameworks)}\n\n"
            f"Synthesize these data infrastructure signals. "
            f"What are the top 5 dataset objects the community should "
            f"build right now? Prioritize by impact on model quality.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )

    fmt_spec = _FORMATS[fmt]
    return {
        "format": fmt,
        "prompt": prompt,
        "min_len": fmt_spec["min_len"],
        "max_tokens": fmt_spec["max_tokens"],
    }


def gen_sovereign_scenario(rng: random.Random, idx: int) -> dict:
    """Generate blockchain/sovereign compute scenario."""
    sub = idx % 6
    topic = rng.choice(_SOVEREIGN_TOPICS)
    gpu = rng.choice(_GPU_ECONOMICS)

    if sub < 3:
        # Standard sovereign analysis (60%)
        fmt = rng.choice(["TREND_ANALYSIS", "DEEP_DIVE", "MARKET_BRIEF"])
        prompt = (
            f"Analyze: {topic}\n\n"
            f"Infrastructure context: {gpu}\n\n"
            f"Cover: what's actually being built (not just whitepapers), "
            f"concrete technical architecture, what works vs what's vaporware, "
            f"the real challenges (latency, verification overhead, coordination), "
            f"and your honest assessment of what's viable in the next 2 years.\n\n"
            f"IMPORTANT: Stay technical and grounded. This is infrastructure analysis, "
            f"not crypto promotion. Separate real projects from token hype.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 3:
        # Signal extraction — what's real in sovereign compute?
        fmt = "SIGNAL_EXTRACTION"
        prompt = (
            f"EVENT: {topic}\n\n"
            f"Infrastructure context: {gpu}\n\n"
            f"Find the weak signal. Beneath the blockchain noise and token hype, "
            f"what is actually shifting in decentralized compute? "
            f"What infrastructure change is real?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 4:
        # Gap detection — what's missing in sovereign AI?
        fmt = "GAP_DETECTION"
        prompt = (
            f"TOPIC: {topic}\n\n"
            f"Infrastructure context: {gpu}\n\n"
            f"What capabilities are missing from sovereign/decentralized AI? "
            f"What data doesn't exist? What would a training dataset "
            f"for decentralized inference optimization look like?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    else:
        # Trend synthesis — top 5 sovereign compute opportunities
        topics = _pick(_SOVEREIGN_TOPICS, rng, 4)
        gpus = _pick(_GPU_ECONOMICS, rng, 2)
        fmt = "TREND_SYNTHESIS"
        prompt = (
            f"INPUT SIGNALS:\n"
            f"- Sovereign compute developments: {'; '.join(topics)}\n"
            f"- GPU economics: {'; '.join(gpus)}\n"
            f"- Blockchain infrastructure: {rng.choice(_SOVEREIGN_TOPICS)}\n\n"
            f"Synthesize these signals. What are the top 5 dataset objects "
            f"that would accelerate decentralized AI infrastructure? "
            f"Separate real opportunity from token hype.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )

    fmt_spec = _FORMATS[fmt]
    return {
        "format": fmt,
        "prompt": prompt,
        "min_len": fmt_spec["min_len"],
        "max_tokens": fmt_spec["max_tokens"],
    }


def gen_culture_scenario(rng: random.Random, idx: int) -> dict:
    """Generate tech culture & builder stories scenario."""
    sub = idx % 5
    if sub == 0:
        # Builder stories
        story = rng.choice(_BUILDER_STORIES)
        topic = rng.choice(_CURATION_TOPICS + _SOVEREIGN_TOPICS)
        fmt = rng.choice(["CULTURE_REPORT", "JOURNAL"])
        prompt = (
            f"Write about: {story}\n\n"
            f"Technical context: {topic}\n\n"
            f"Cover: specific examples and names where possible, "
            f"what these builders are actually shipping, "
            f"the real challenges they face, "
            f"what the broader community can learn, "
            f"and whether this trend is growing or plateauing.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 1:
        # Ethics and debates
        debate = rng.choice(_ETHICS_DEBATES)
        culture = rng.choice(_BUILDER_STORIES)
        fmt = rng.choice(["CULTURE_REPORT", "JOURNAL"])
        prompt = (
            f"Analyze this ongoing debate: {debate}\n\n"
            f"Community context: {culture}\n\n"
            f"Cover: the strongest arguments on each side (steelman both), "
            f"what the actual evidence says (not vibes), "
            f"where the discourse is productive vs performative, "
            f"and your take on what matters and what's noise.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 2:
        # Signal extraction — what cultural shift matters?
        story = rng.choice(_BUILDER_STORIES)
        debate = rng.choice(_ETHICS_DEBATES)
        fmt = "SIGNAL_EXTRACTION"
        prompt = (
            f"EVENT: {story}\n\n"
            f"Cultural tension: {debate}\n\n"
            f"Find the weak signal. What cultural shift in AI is "
            f"actually happening beneath the discourse? "
            f"What will this look like in 2 years?\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    elif sub == 3:
        # Gap detection — what's missing in AI culture discourse?
        debate = rng.choice(_ETHICS_DEBATES)
        story = rng.choice(_BUILDER_STORIES)
        fmt = "GAP_DETECTION"
        prompt = (
            f"TOPIC: {debate}\n\n"
            f"Builder context: {story}\n\n"
            f"What perspectives are missing from this debate? "
            f"What data would ground the conversation? "
            f"Define a dataset that captures what discourse currently ignores.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )
    else:
        # Trend synthesis — top 5 from culture signals
        stories = _pick(_BUILDER_STORIES, rng, 3)
        debates = _pick(_ETHICS_DEBATES, rng, 3)
        fmt = "TREND_SYNTHESIS"
        prompt = (
            f"INPUT SIGNALS:\n"
            f"- Builder trends: {'; '.join(stories)}\n"
            f"- Active debates: {'; '.join(debates)}\n"
            f"- Technical context: {rng.choice(_CURATION_TOPICS)}\n\n"
            f"Synthesize these cultural signals. What are the top 5 "
            f"dataset opportunities that would help AI models understand "
            f"the human side of technology? Think: builder decision-making, "
            f"community dynamics, ethical reasoning.\n\n"
            f"{_FORMATS[fmt]['instruction']}"
        )

    fmt_spec = _FORMATS[fmt]
    return {
        "format": fmt,
        "prompt": prompt,
        "min_len": fmt_spec["min_len"],
        "max_tokens": fmt_spec["max_tokens"],
    }


CATEGORY_GENERATORS = {
    "infra":     gen_infra_scenario,
    "industry":  gen_industry_scenario,
    "curation":  gen_curation_scenario,
    "sovereign": gen_sovereign_scenario,
    "culture":   gen_culture_scenario,
}

# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

session = requests.Session()
api_lock = threading.Lock()
api_calls = Counter()


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
            # Strip <think>...</think> blocks
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
# QUALITY CHECK — v2: stricter anti-hype, technical specificity
# ═══════════════════════════════════════════════════════════════════════════════

_DEGEN_PATTERN = re.compile(r"(.{40,})\1{2,}")

# Expanded corporate speak / hype detector
_CORP_SPEAK = re.compile(
    r"\b(leverag(?:e|ing)|synerg(?:y|ies)|paradigm shift|game[- ]chang(?:ing|er)|"
    r"revolutionary|disruptive innovation|move the needle|circle back|"
    r"best[- ]in[- ]class|world[- ]class|cutting[- ]edge|groundbreaking|"
    r"unlock(?:ing)? (?:the )?potential|at the end of the day|"
    r"going forward|in terms of|it goes without saying|"
    r"in this article we will|without further ado|"
    r"let's dive (?:in|deep)|buckle up|strap in|"
    r"the future is here|mark my words|"
    r"first and foremost|needless to say|"
    r"it's worth noting that|interestingly enough)\b",
    re.IGNORECASE,
)

# Generic blogging detector
_GENERIC_BLOG = re.compile(
    r"\b(in today's rapidly (?:evolving|changing)|"
    r"in recent years|as we all know|"
    r"it's no secret that|the landscape is shifting|"
    r"the question (?:remains|is)|have you ever wondered|"
    r"imagine a world where|picture this)\b",
    re.IGNORECASE,
)

# Technical specificity — real analysis has concrete details
_SPECIFICS = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|"  # proper nouns
    r"\$[\d,.]+[BMK]?|"                        # dollar amounts
    r"\d+\.?\d*\s*%|"                          # percentages
    r"\d+\.?\d*[BMK]\b|"                       # quantities with B/M/K
    r"\d{4}\b|"                                 # years
    r"\d+\.?\d*\s*(?:GB|TB|MB|VRAM|tok/s|tokens?|params?|layers?|heads?)",  # technical units
    re.MULTILINE,
)


def quality_check(content: str, category: str) -> tuple[bool, str]:
    """Check if gen output meets v2 quality: technical journalism, no hype."""
    min_lens = {
        "infra": 800, "industry": 600, "curation": 800,
        "sovereign": 600, "culture": 500,
    }
    min_len = min_lens.get(category, 600)
    if len(content) < min_len:
        return False, f"too_short_{len(content)}/{min_len}"

    if _DEGEN_PATTERN.search(content):
        return False, "degenerate"

    # Stricter: zero corporate speak tolerance
    corp_hits = _CORP_SPEAK.findall(content)
    if len(corp_hits) > 1:
        return False, f"corporate_speak_{len(corp_hits)}"

    # Generic blogging detector
    generic_hits = _GENERIC_BLOG.findall(content)
    if len(generic_hits) > 0:
        return False, f"generic_blog_{len(generic_hits)}"

    # Technical specificity — v2 requires more concrete details
    specifics = _SPECIFICS.findall(content)
    min_specifics = {"infra": 12, "industry": 10, "curation": 10,
                     "sovereign": 8, "culture": 6}
    required = min_specifics.get(category, 8)
    if len(specifics) < required:
        return False, f"vague_{len(specifics)}/{required}"

    # Section structure — all formats should have structure
    sections = len(re.findall(r"(?:^|\n)(?:#{1,3}\s|(?:\d+\.|\*|\-)\s)", content))
    if sections < 2:
        return False, f"no_structure_{sections}"

    return True, "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING & PROGRESS — same as v1
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
            "version": "v2",
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
# PAIR GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def grind_pair(category: str, scenario: dict, idx: int) -> dict | None:
    """Generate one training pair. Maverick gen → quality gate → rewrite."""
    user_msg = scenario["prompt"]
    max_tokens = scenario.get("max_tokens", 3072)
    min_len = scenario.get("min_len", 500)

    with api_lock:
        api_calls["gen"] += 1
    content = together_call(SYSTEM_PROMPT, user_msg, model=GEN_MODEL,
                            max_tokens=max_tokens, min_len=min_len // 2)
    if not content:
        return None

    passed, reason = quality_check(content, category)
    tier = "gen"

    if not passed:
        with api_lock:
            api_calls["pass"] += 1
        content = together_call(SYSTEM_PROMPT, user_msg, model=PASS_MODEL,
                                max_tokens=max_tokens, min_len=min_len // 2)
        if not content:
            return None
        tier = "rewrite"

    fp = hashlib.sha256(f"v2-{category}-{idx}-{content[:200]}".encode()).hexdigest()[:16]

    return {
        "id": f"swarmsignal-v2-{category}-{idx:06d}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": content},
        ],
        "metadata": {
            "domain": "ai_signal",
            "version": "v2",
            "category": category,
            "format": scenario["format"],
            "model": GEN_MODEL if tier == "gen" else PASS_MODEL,
            "tier": tier,
            "fingerprint": fp,
            "source": "swarmsignal-cook-v2",
            "cooked_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY GRINDER
# ═══════════════════════════════════════════════════════════════════════════════

def grind_category(category: str, target: int | None = None, seed: int = SEED):
    """Grind one category with parallel workers."""
    generator = CATEGORY_GENERATORS[category]
    target = target or CATEGORY_TARGETS[category]

    out_file = OUTPUT_DIR / f"cat_{category}.jsonl"

    print(f"\n{'='*70}")
    print(f"  CATEGORY: {category.upper()} — {target:,} pairs")
    print(f"  Gen: {GEN_MODEL}")
    print(f"  Pass: {PASS_MODEL}")
    print(f"  Workers: {WORKERS}")
    print(f"{'='*70}")

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

    print(f"  Testing Gen model...")
    test1 = together_call("You are SwarmSignal.", "Say 'ready'.",
                          model=GEN_MODEL, max_tokens=10, min_len=1)
    if not test1:
        print(f"  FAIL — Gen model not responding")
        return written
    print(f"  Gen: OK | Pass: OK (same model)")

    t0 = time.time()
    session_start = written

    work_queue = []
    for idx in range(target * 2):
        pair_id = f"{category}-{idx}"
        if pair_id in done_ids:
            continue
        rng = random.Random(seed + idx)
        scenario = generator(rng, idx)
        work_queue.append((idx, scenario))
        if written + len(work_queue) >= target + (target // 5):
            break

    print(f"  Work queue: {len(work_queue):,} tasks")

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

    save_checkpoint(category, written, list(done_ids)[-10000:])
    update_progress(category, written, target, gen_pass, rewritten, failed)

    elapsed = time.time() - t0
    rate = (written - session_start) / max(elapsed / 60, 0.1)
    rw_pct = rewritten / max(gen_pass + rewritten, 1) * 100
    print(f"\n  Category {category} done: {written:,} pairs, {rate:.0f}/min, "
          f"{elapsed/60:.1f} min")
    print(f"  Gen pass: {gen_pass:,} | Rewritten: {rewritten:,} ({rw_pct:.0f}%)")

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def assemble():
    """Merge all category files into final output."""
    print(f"\n{'='*70}")
    print(f"  ASSEMBLY — v2")
    final = OUTPUT_DIR / "swarmsignal_v2_30k.jsonl"
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
    print(f"  Gen pass: {gen_total:,} | Rewrites: {rw_total:,} ({rw_pct:.0f}%)")
    print(f"  Output: {final}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def show_status():
    if PROGRESS_FILE.exists():
        p = json.loads(PROGRESS_FILE.read_text())
        print(f"\nSwarmSignal Cook v2 — Progress")
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
        description="SwarmSignal Cook v2 — Technical Journalism Expansion (30K)")
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

    api_key = os.environ.get("TOGETHER_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set TOGETHER_KEY environment variable")
        sys.exit(1)

    if api_key:
        init_session(api_key)

    WORKERS = args.workers

    print(f"{'='*70}")
    print(f"  SWARMSIGNAL COOK v2 — 30K Technical Journalism Expansion")
    print(f"  Gen/Pass: {GEN_MODEL}")
    print(f"  Workers:  {WORKERS}")
    print(f"  Seed:     {args.seed}")
    print(f"  Categories:")
    for cat, target in CATEGORY_TARGETS.items():
        print(f"    {cat:<15} {target:>6,} pairs")
    print(f"  Total:    {TOTAL_TARGET:,} pairs")
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
                print(f"  {scenario['prompt'][:400]}...")
                print(f"  min_len={scenario['min_len']}, max_tokens={scenario.get('max_tokens', 3072)}")
        return

    _progress["start_time"] = time.time()

    cats_to_run = (CATEGORY_TARGETS if args.category == "all"
                   else {args.category: CATEGORY_TARGETS[args.category]})
    for cat, default_target in cats_to_run.items():
        target = args.target or default_target
        grind_category(cat, target=target, seed=args.seed)

    assemble()

    rw_rate = api_calls.get("pass", 0) / max(api_calls.get("gen", 0), 1) * 100
    print(f"\n{'='*70}")
    print(f"  DONE — SwarmSignal Cook v2 Complete")
    print(f"  API calls: {api_calls['total']:,}")
    print(f"    Gen:     {api_calls.get('gen', 0):,}")
    print(f"    Rewrite: {api_calls.get('pass', 0):,} ({rw_rate:.0f}%)")
    print(f"    Errors:  {api_calls.get('error', 0)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
