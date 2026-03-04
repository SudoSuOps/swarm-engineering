# SwarmSignal — AI Market Intelligence Writer

**SwarmSignal-9B** is a fine-tuned AI model that writes sharp, opinionated market intelligence about the AI industry. Built by [Swarm & Bee](https://swarmandbee.ai). No corporate fluff. Real analysis.

```
Base:     Qwen/Qwen3.5-9B (Mamba-Transformer hybrid, 9.5B params)
Method:   bf16 LoRA r=64 (full precision, no quantization during training)
Data:     30,000 AI trends writer pairs (28.5K train + 1.5K eval)
GPU:      RTX PRO 6000 Blackwell 96GB
Training: 891 steps, 3h21m, final loss 0.4031
GGUF:     Q4_K_M — 5.3GB, runs on any 8GB+ GPU
```

## What It Writes

Six output formats, each with a distinct editorial voice:

| Format | Description | Length |
|--------|-------------|--------|
| **BRIEFING** | Daily intel — 5-7 stories with analysis, sentiment score, trend callouts, sleeper story | 800-1200 words |
| **JOURNAL** | Weekly synthesis — deep dives, vibe check, contrarian take, predictions | 2000-3000 words |
| **TREND_ANALYSIS** | Single trend deep dive — what, why, winners/losers, outlook | 1200-1800 words |
| **COMPANY_PROFILE** | Company move analysis — signals, implications, honest take | 800-1200 words |
| **CULTURE_REPORT** | Community discourse — what devs actually talk about, vibes, drama | 600-1000 words |
| **MARKET_BRIEF** | Funding & M&A — follow the money, valuation trends | 600-1000 words |

## Quick Start

### Run with llama-server (recommended)

```bash
# Serve the Q4_K_M GGUF
llama-server -m swarmsignal-9b-q4km.gguf -ngl 99 -c 4096 --port 8090

# Generate a briefing
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are SwarmSignal, an AI market intelligence analyst and writer for Swarm & Bee."},
      {"role": "user", "content": "BRIEFING: Cover today'\''s top AI stories."}
    ],
    "max_tokens": 2048,
    "temperature": 0.7
  }'
```

### Daily Pipeline (automated)

```bash
# 1. Collect live signals from RSS, arXiv, Hacker News
python3 -m src.collect_signals

# 2. Generate daily briefing using the fine-tuned model
python3 -m src.generate_daily --backend llama --format briefing

# 3. Generate weekly journal (Fridays)
python3 -m src.generate_daily --backend llama --format journal
```

## Project Structure

```
Swarm-Signal/
├── src/
│   ├── cook_swarmsignal.py      # Data factory — 30K pairs via Together.ai
│   ├── assemble_swarmsignal.py  # Train/eval split assembler
│   ├── train_swarmsignal_9b.py  # Blackwell training script (Unsloth + TRL)
│   ├── collect_signals.py       # Live RSS + arXiv + HN signal collector
│   └── generate_daily.py        # Daily report generator (llama-server/Ollama)
├── eval/
│   └── sanity_prompts.json      # Evaluation fixtures
├── scripts/
│   ├── serve.sh                 # Launch llama-server
│   └── daily_pipeline.sh        # Cron-ready daily signal pipeline
├── docs/
│   └── MODEL_CARD.md            # Full model card with training details
└── README.md
```

## Data Factory

The training data was cooked using a two-tier architecture on Together.ai:

```
Tier 1 (GEN):  Llama-4-Maverick-17B-128E — fast generation
Tier 2 (PASS): Same model — quality rewrite for failures

6 Categories:
  BRIEFING   10,000 pairs    daily intel
  TRENDS      8,000 pairs    deep dives
  COMPANY     4,000 pairs    launches, pivots, M&A
  CULTURE     4,000 pairs    community discourse
  MARKET      2,000 pairs    funding & valuations
  JOURNAL     2,000 pairs    weekly synthesis
```

Quality gates enforce:
- No degeneration (repeated text)
- No corporate speak ("leveraging", "synergies", "paradigm shift")
- Minimum specificity (proper nouns, dollar amounts, percentages, years)
- Section structure for longer formats

```bash
# Cook new data (requires Together.ai API key)
TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal

# Check progress
TOGETHER_KEY=tgp_v1_... python3 -m src.cook_swarmsignal --status

# Dry run (preview prompts)
python3 -m src.cook_swarmsignal --dry-run

# Assemble train/eval splits
python3 -m src.assemble_swarmsignal
```

## Training

Trained on a single RTX PRO 6000 Blackwell (96GB) using Unsloth + TRL:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
WANDB_MODE=disabled python3 src/train_swarmsignal_9b.py
```

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3.5-9B |
| LoRA rank | 64 |
| LoRA alpha | 32 |
| Precision | bf16 (no quantization) |
| Batch (effective) | 4 × 8 = 32 |
| Max sequence | 4096 |
| Learning rate | 5e-5, cosine schedule |
| NEFTune alpha | 5.0 |
| Packing | True |
| Steps | 891 |
| Time | 3h 21m |
| Final loss | 0.4031 |

## Model Artifacts

| Artifact | Size | SHA256 |
|----------|------|--------|
| Merged bf16 (4 shards) | 18 GB | — |
| FP16 GGUF | 17.9 GB | — |
| Q4_K_M GGUF | 5.3 GB | `fde9dc8fb625adfb91d35e6908c5ef65be0543c1043b13bcb24b25580f366560` |
| LoRA adapter | 465 MB | — |

## Voice

SwarmSignal writes like a senior analyst with opinions:

> *"NVIDIA's Blackwell Ultra GPU, shipping with 288GB HBM3e, is the most significant memory upgrade in the datacenter since 2018. This isn't a tweak; it's a density redefinition."*

> *"OpenAI's $40B funding round at a $340B valuation is a market bet on general intelligence, not just NLP."*

> *"Meta is segmenting its Llama 4 output into 'Scout' and 'Maverick'. This is a direct challenge to the OpenAI pricing model."*

## Requirements

```
requests>=2.31.0
unsloth>=2026.2.0     # training only
transformers>=5.0.0   # training only
trl>=0.24.0           # training only
datasets>=4.0.0       # training only
torch>=2.0.0          # training only
```

## Links

- **Live**: [swarmandbee.ai/signal](https://swarmandbee.ai/signal)
- **HuggingFace**: [SwarmandBee](https://huggingface.co/SwarmandBee)
- **GitHub**: [SudoSuOps](https://github.com/SudoSuOps)
- **Discord**: [Swarm & Bee](https://discord.gg/sHXNzNhscc)

## License

Proprietary. Copyright 2026 Swarm & Bee. All rights reserved.
