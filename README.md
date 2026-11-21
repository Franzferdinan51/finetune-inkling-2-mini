# finetune-inkling-2-mini

**Fine-tune Inkling 2 Mini on Tinker, then serve it in LM Studio or publish it.** Adapted from Migel Tissera's Inkling 2 Small recipe — the one behind Tess-4.5-594B-A17B.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-agentskills.io-black)](https://agentskills.io)
[![Works with](https://img.shields.io/badge/works%20with-Claude%20Code%20%C2%B7%20Codex%20%C2%B7%20OpenCode-brightgreen)](#install-the-skill)
[![Model](https://img.shields.io/badge/model-Inkling%202%20Mini-7b3fe4)](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

Inkling 2 Mini is a 26B-parameter Mixture-of-Experts model with 5.4B active
parameters, a million-token window and effort-controlled reasoning. You can
fine-tune it on Tinker in an afternoon. Getting a fine-tune that behaves well, works
at long context, and runs outside Tinker when you want it to has traps that cost
real runs:
- a tokenizer that silently changed,
- experts stored in 4-bit that a normal LoRA cannot be merged into,
- a 64K cap that counts output tokens,
- a reasoning effort that moved the benchmark score by 18 points on the original run.

This skill hands your coding agent the whole path, with each of those decisions
made explicit and every step scripted. Install it and say *"fine-tune Inkling 2
Mini on data/rows.jsonl"*. The agent takes it from there, and stops before anything
that costs money. Keep the model private on Tinker, serve it in LM Studio, or
publish it: the skill covers each, and nothing is uploaded unless you ask.

## Provenance

This skill is adapted from the Inkling 2 Small skill by Migel Tissera, the exact
method that produced [Tess-4.5-594B-A17B](https://huggingface.co/migtissera/Tess-4.5-594B-A17B)
(88.2% on SWE-Bench Pro v2 Hard at xhigh, one epoch, 46.9M tokens, an exact merge
verified token by token). The workflow, traps and discipline are his; this version
substitutes Inkling 2 Mini's own facts, Tinker prices and an LM Studio-first
serving path. Worked numbers quoted in the references are that Small run's, and
are labelled as such — re-measure on Mini.

## What makes Inkling 2 Mini different, and what the skill does about it

| The trap | What happens | What the skill does |
|---|---|---|
| **Tokenizer.** Inkling 2 uses H200K; earlier Inklings use o200k | An old renderer package used the wrong one without an error; "64K" rows overflowed by up to 16K tokens | One shared rendering path that asserts H200K; every row measured with it |
| **4-bit experts.** The open checkpoint stores experts in NVFP4 (modelopt) | A LoRA on the experts cannot be merged without lossy requantization | `--adapter release`: attention + unembed LoRA, merged exactly in FP32; a merge script that refuses anything else |
| **Caps count output.** 65,536 / 262,144 on Tinker | An agent's first request plus 16K of output room is rejected | Train at 64K, transplant the adapter to 256K in seconds, no retraining |
| **Agent data says "start working".** 112 of 113 sessions in the original run's data opened with a tool call | The fine-tune calls tools on "Hey there!" | A self-distilled conversational slice from the base, and a probe for every checkpoint |
| **Effort is a dial.** Rows are trained at one effort | 70.6% at `high`, 88.2% at `xhigh` on the Small run — same model, same tasks | Effort fixed in training, swept in evaluation |
| **"Lossless" is a claim** | A wrong scale or a skipped shard looks fine until it doesn't | Per-token logprob comparison of the served merge against the Tinker adapter |

## The workflow

| Step | What happens | Script |
|---|---|---|
| 1. Rows | `{"messages", "tools", "group"}` JSONL, from [tinker-ai-skill](https://github.com/drost-ai/tinker-ai-skill) or your own pipeline | `make_synthetic_rows.py` for a dry run |
| 2. Prepare | Normalize, render every row with the training rendering, enforce the budget, split by session, report cost and provenance | `prepare_dataset.py` |
| 3. Converse | Sample a conversational slice from the base so the model keeps talking when it should | `conversational_slice.py` |
| 4. Train | LoRA on Tinker with the chosen adapter shape, smoke test and cost first, W&B on | `train.py` |
| 5. Probe | Greeting, no-tool question, task: on the base and every checkpoint | `probe.py` |
| 6. 256K | Move the adapter to the 256K variant, needle-test it | `transplant_256k.py` |
| 7. Merge | Exact merge into the released checkpoint, locally or on Modal | `merge_lossless.py`, `modal_app.py` |
| 8. Verify | Token-level comparison of the served merge against the adapter | `merge_check.py` |
| 9. Serve | LM Studio first (GGUF or MLX, DSpark drafter); SGLang/vLLM as the datacenter option | `references/export-merge-serve.md` |
| 10. Evaluate | Base on your harness first, effort swept | `references/evaluation.md` |

## Quick start: five minutes, no data, no spend

```
git clone https://github.com/Franzferdinan51/finetune-inkling-2-mini.git ~/.claude/skills/finetune-inkling-2-mini
cd ~/.claude/skills/finetune-inkling-2-mini
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

python scripts/make_synthetic_rows.py --out demo/rows.jsonl --sessions 40 --long-results
python scripts/prepare_dataset.py --input demo/rows.jsonl --name demo --out-dir demo/datasets --val-fraction 0.1
python scripts/train.py --dataset demo/datasets/demo          # offline smoke test: decoded row, steps, cost
python scripts/merge_lossless.py selftest                      # the exact-merge guarantees, on a toy checkpoint
```

Then put your keys in `.env`, point step 2 at your own rows, and let your agent
follow `SKILL.md`.

## Install the skill

Clone it into your agent's skills directory. The directory must be named
`finetune-inkling-2-mini`, which is what `git clone` gives you by default:

| Agent | Personal (all projects) | Project (this repo) |
|---|---|---|
| Claude Code | `~/.claude/skills/finetune-inkling-2-mini/` | `.claude/skills/finetune-inkling-2-mini/` |
| OpenAI Codex | `~/.agents/skills/finetune-inkling-2-mini/` | `.agents/skills/finetune-inkling-2-mini/` |
| OpenCode | `~/.config/opencode/skills/finetune-inkling-2-mini/` | `.opencode/skills/finetune-inkling-2-mini/` |

```
git clone https://github.com/Franzferdinan51/finetune-inkling-2-mini.git ~/.claude/skills/finetune-inkling-2-mini
git clone https://github.com/Franzferdinan51/finetune-inkling-2-mini.git ~/.agents/skills/finetune-inkling-2-mini
git clone https://github.com/Franzferdinan51/finetune-inkling-2-mini.git ~/.config/opencode/skills/finetune-inkling-2-mini
```

Invoke it with `/finetune-inkling-2-mini` in Claude Code; Codex and OpenCode pick it
up from its description. Any agent that supports [Agent Skills](https://agentskills.io)
works the same way.

## What it costs

Tinker list prices, November 2026
([current prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)):

| | Cost |
|---|---|
| Training, 64K variant | $0.75 per million tokens: ~1,000 rows of up to 64K ≈ 47M tokens ≈ **$35 per epoch** |
| Training, 256K variant | $1.50 per million tokens (rarely needed: transplant instead) |
| Conversational slice, 150 conversations | a few dollars |
| Probes, transplant, needle test | cents |
| Exact merge | CPU only: minutes of compute plus ~60 GB of disk for a few hours |
| Serving | LM Studio on one consumer GPU or Apple Silicon: the Q4_K_M GGUF is 17.5 GB; MLX 4-bit also loads |

The dataset steps, smoke tests and the merge self-test are free and run on a laptop.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip)
- `tinker-cookbook>=0.6.0`, `tml-renderers>=0.4.0` (see `requirements.txt`)
- A [Tinker](https://tinker.thinkingmachines.ai) API key; [Weights & Biases](https://wandb.ai/authorize) optional
- To publish: a Hugging Face token with write access; to merge: ~60 GB of disk or a
  [Modal](https://modal.com) account; to self-host: LM Studio and ~20 GB of GPU/RAM

## Layout

```
SKILL.md                          the workflow your agent follows
scripts/inkling2.py               the one rendering path, constants, row normalization
scripts/make_synthetic_rows.py    step 1: synthetic rows for a dry run
scripts/prepare_dataset.py        step 2: validate, measure, split, report
scripts/conversational_slice.py   step 3: self-distilled conversational rows
scripts/train.py                  step 4: LoRA on Tinker (release / full adapter)
scripts/probe.py                  step 5: behaviour probe, any OpenAI-compatible endpoint
scripts/transplant_256k.py        step 6: 64K adapter onto the 256K variant, needle test
scripts/merge_lossless.py         step 7: exact merge, check, selftest
scripts/modal_app.py              step 7 on Modal: download, merge, upload, serve
scripts/merge_check.py            step 8: token-level verification of the merge
references/                       model facts, adapter shape, training, behaviour,
                                  long context, export/merge/serve, evaluation, lessons
```

## Author and license

Written by [Migel Tissera](https://huggingface.co/migtissera), creator of the Tess
models. MIT license. Inkling 2 Mini, Tinker and tinker-cookbook are products of
Thinking Machines Lab; this skill is an independent guide built on them.

Adapted for Inkling 2 Mini (LM Studio-first serving, Mini's facts and Tinker
prices) by [Duckets](https://github.com/Franzferdinan51).

If this saved you a run, star the repository and share what you trained.
