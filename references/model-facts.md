# Inkling 2 Mini: the facts that change how you fine-tune it

Checked against the model card, the cookbook and our own runs in November 2026.
Prices and service limits move; re-check the linked pages before quoting them.

## The model

| | |
|---|---|
| Name | `thinkingmachines/Inkling-2-Mini` (Thinking Machines Lab), Apache 2.0 |
| Shape | Mixture of Experts (`inkling2_loop`), 26B total / 5.4B active parameters, 24 layers, 128 routed experts (top-8), hidden size 2304 |
| Context | 1,048,576 tokens natively |
| Thinking | Effort-conditioned. `reasoning_effort` takes a preset (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) or a float; `high` is 0.9, the renderer's default |
| Inputs | Text and images. This skill covers text fine-tuning only |
| Extras | A built-in MTP head (3 layers), and a DSpark draft model for speculative decoding (a separate GGUF in the unsloth repo) |

## Tokenizer and renderer

- Renderer: `tml_v0`, from the `tml-renderers` package that `tinker-cookbook` pulls in.
  Resolve it with `model_info.get_recommended_renderer_name(model)`; never hardcode it.
- Tokenizer: **H200K** (`H200kBaseChatTokenizer`). Inkling and Inkling-Small use the
  older o200k vocabulary. `tml-renderers` 0.1.0 rendered Inkling 2 with o200k
  *without any error*; rows chunked to 64K came out shorter than reality and 77 of
  834 overflowed when rendered correctly (the longest was 81,460 tokens). Pin
  `tml-renderers>=0.4.0` and `tinker-cookbook>=0.6.0`; `scripts/inkling2.py` refuses
  to render with any tokenizer but H200K.
- What a rendered row looks like: system prompt, then a zero-loss `tool_declare`
  system message carrying the tool JSON, then a system line `Thinking effort level:
  0.9`, then the conversation. Assistant turns are `<|message_model|>` messages with
  `<|content_thinking|>`, text and `<|content_invoke_tool_json|>` parts.
- There is no special think token, so literal `<think>` text in your data is just
  text (unlike Qwen and Nemotron, where it tokenizes as control tokens).
- The renderer keeps earlier-turn thinking in history (the "extension property"), so
  the cookbook trains a whole multi-turn row as one example. No preserve-thinking
  variant is needed.

## On Tinker

| Tinker model name | Hard cap (prompt + output) | Train price per 1M tokens |
|---|---|---|
| `thinkingmachines/Inkling-2-Mini` | 65,536 | $0.75 |
| `thinkingmachines/Inkling-2-Mini:peft:262144` | 262,144 | $1.50 |

Sampling: $0.15 per 1M prompt tokens, $0.015 cached, $0.50 per 1M output tokens.
Prices: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/

- No Tinker endpoint serves the 1M window. A client configured for 1M against the
  plain model name only *appears* to work until a request crosses 65,536.
- The cap counts output. A 50K prompt with `max_tokens` 16,384 is rejected on the
  64K variant ("rejected the request as invalid").
- Tinker's OpenAI-compatible (`.../oai/api/v1`) and Anthropic-compatible
  (`.../anthropic/api/v1`) endpoints rendered our requests identically (prompt token
  counts within 12) and both keep earlier-turn thinking for Inkling, so no proxy is
  needed to serve the way you trained. Sampler paths (`tinker://.../sampler_weights/<name>`)
  work as model names.
- `hyperparam_utils.get_lr` raises `NotImplementedError` for Inkling models. You
  choose the learning rate (`references/training.md`).
- Periodic checkpoints carry the cookbook's TTL (7 days); `final` is kept. Point
  long-lived clients at a TTL-free sampler path (`final`, or a transplant you saved).
  Sampler checkpoints cannot be downloaded; download `weights/<name>`.

## The released checkpoint

Open weights came out after the model had been on Tinker for about a week. What
matters for fine-tuners is the storage format:

| Tensors | Format |
|---|---|
| Routed experts | NVFP4 (the config's `quantization_config` is modelopt / NVFP4) |
| Attention projections, router, embeddings, unembedding head | BF16 |

About 19.5 GB in a single safetensors shard (`model_type: inkling2_loop`). Which
modules carry exact BF16 targets is what `merge_lossless.py check` enumerates
against the real index — run it before trusting any table, this one included.
Serving the checkpoint as-is needs one 80 GB-class GPU; the quantised builds below
need far less.

This table is why `references/adapter-shape.md` exists: a LoRA on the experts cannot
be merged into 4-bit weights without requantizing them.

## GGUF and MLX exist for Mini

Unlike Inkling 2 Small, Mini's architecture (`inkling2_loop`) is supported by
llama.cpp, so the local route is real — and it is this skill's preferred serve path:

- **GGUF**: `lmstudio-community/Inkling-2-Mini-GGUF` ships Q4_K_M (17.5 GB), Q6_K
  and Q8_0, plus a BF16 mmproj for the vision tower. The unsloth GGUF repo adds a
  UD-Q4_K_XL and the DSpark drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`).
- **MLX**: 4-bit builds exist (affine, group size 64 — the Unsloth-style recipe),
  e.g. `mlx-community/Inkling-2-Mini-4bit`.
- **LM Studio** loads both formats natively; serve on its OpenAI-compatible
  endpoint (`references/export-merge-serve.md`). SGLang/vLLM and Tinker remain
  options for datacenter or no-GPU serving.
