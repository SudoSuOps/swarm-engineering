# SwarmSignal-9B-v1 — Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Name** | SwarmSignal-9B-v1 |
| **Base model** | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| **Architecture** | Mamba-Transformer hybrid (GDN), 9.5B parameters |
| **Method** | bf16 LoRA r=64, alpha=32, no quantization |
| **Domain** | AI market intelligence writing |
| **Training GPU** | NVIDIA RTX PRO 6000 Blackwell (96GB VRAM) |
| **Framework** | Unsloth 2026.2.1 + TRL 0.24.0 + PyTorch 2.9.1+cu128 |
| **License** | Proprietary — Swarm & Bee |

## Training Data

- **Source**: Together.ai two-tier cook (Llama-4-Maverick-17B-128E gen + quality rewrite)
- **Total pairs**: 30,000 (28,500 train + 1,500 eval)
- **Train SHA256**: `358c5275e03e8553ca1cfc558b6468391a64f622822ff0facd521d44aff3f888`
- **Categories**: briefing (10K), trends (8K), company (4K), culture (4K), market (2K), journal (2K)

### Quality Gates

Every training pair passed:
1. **Length check** — minimum chars per category (600-1500)
2. **Degeneration detector** — rejects repeated sentence patterns
3. **Corporate speak filter** — max 2 violations of banned phrases
4. **Specificity scoring** — minimum proper nouns, dollar amounts, percentages
5. **Section structure check** — briefings and journals require 3+ sections

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Precision | bf16 (full precision, no 4-bit) |
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Max sequence length | 4096 |
| Learning rate | 5e-5 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| NEFTune alpha | 5.0 |
| Packing | True |
| Epochs | 1 |
| Seed | 42 |

## Training Results

| Metric | Value |
|--------|-------|
| Total steps | 891 |
| Training time | 3h 21m (12,090s) |
| Final train loss | 0.4031 |
| Final step loss | ~0.333 |
| Final grad norm | ~0.15 |
| Samples/second | 2.357 |
| Steps/second | 0.074 |

Loss trajectory: rapid descent from ~2.0 to ~0.5 in first 100 steps, gradual convergence to ~0.33 by step 891.

## Quantization

| Format | Size | BPW | SHA256 |
|--------|------|-----|--------|
| bf16 merged | 18 GB | 16.0 | — |
| FP16 GGUF | 17.9 GB | 16.0 | — |
| Q4_K_M GGUF | 5.3 GB | 5.02 | `fde9dc8f...` |
| LoRA adapter | 465 MB | — | — |

Q4_K_M quantized via llama.cpp `llama-quantize`. Zero observable quality loss in eval testing.

## Inference Performance (Q4_K_M)

Tested on RTX 3090 Ti (24GB, sm_86) via llama-server:

| Metric | Value |
|--------|-------|
| VRAM usage | 5,640 MiB |
| Generation speed | ~108 tok/s |
| Prompt eval speed | ~2,500 tok/s |
| Think mode | Active (reasoning_content) |

## Intended Use

SwarmSignal-9B generates AI market intelligence across six formats:
- Daily briefings with sentiment scoring
- Weekly journal synthesis
- Trend analysis deep dives
- Company/product profiles
- Community culture reports
- Funding and M&A market briefs

## Limitations

- Training data was synthetically generated (not from real news events)
- Dates and specific events referenced in outputs may be fabricated
- Model has strong opinions by design — outputs should be reviewed for accuracy
- Not trained for general-purpose chat or instruction following
- Best results with the exact system prompt used during training

## System Prompt

The model was trained with a specific system prompt (see `src/cook_swarmsignal.py` SYSTEM_PROMPT). Using the same prompt at inference time produces the best results.

## Provenance

```json
{
  "model_name": "SwarmSignal-9B-v1",
  "base_model": "Qwen/Qwen3.5-9B",
  "method": "bf16 LoRA r=64 (full precision)",
  "gpu": "RTX PRO 6000 Blackwell 96GB",
  "cook_source": "Together.ai two-tier (80B gen + 235B pass)",
  "domain": "ai_signal",
  "completed_utc": "2026-03-04T14:03:29.815536"
}
```

## Citation

```
SwarmSignal-9B-v1. Swarm & Bee, 2026.
https://github.com/SudoSuOps/Swarm-Signal
```
