# Export, exact merge, upload, serve

Training leaves an adapter in Tinker storage. Getting a model anyone can run is:
download the adapter, download the base, merge exactly, verify, upload, serve.

## 1. Download the adapter

```
tinker checkpoint download tinker://<run-id>:train:0/weights/final -o ./adapters
```

- Use the **training** path (`weights/<name>`). Sampler checkpoints cannot be
  downloaded. The CLI exports through a temporary external-weights checkpoint
  (it briefly starts a training session on the base) and deletes it afterwards.
- Periodic checkpoints expire after 7 days; `final` does not. Pick by probe and
  eval, then download promptly.
- The directory holds `adapter_model.safetensors` (fp32, PEFT names) and
  `adapter_config.json` (`r`, `lora_alpha`). Files of an age-encrypted checkpoint keep
  a `.age` suffix: decrypt with `age -d` first.

Check it before downloading 19.5 GB of base:

```
hf download thinkingmachines/Inkling-2-Mini model.safetensors.index.json --local-dir ./base
python scripts/merge_lossless.py check --adapter-dir ./adapters/<dir> --base-index ./base/model.safetensors.index.json
```

`check` exits non-zero if any module has no exact BF16 target (an expert adapter).

## 2. Publish the adapter alone (optional, minutes)

```
tinker checkpoint push-hf tinker://<run-id>:train:0/weights/final --repo you/my-model-LoRA
```

A PEFT repo (a few hundred MB) for people who already serve Inkling 2 Mini and want to
switch adapters per request, or re-merge themselves. Private unless `--public`.

## 3. Download the base and merge

Pin the base commit so the merge is reproducible:

```
hf download thinkingmachines/Inkling-2-Mini --revision <commit> --local-dir ./base
python scripts/merge_lossless.py merge --base-dir ./base --adapter-dir ./adapters/<dir> --out-dir ./merged
```

Resources: ~60 GB of disk (base + merged), RAM for the shard plus its copy (32 GB
is comfortable), CPUs only. Mini's base is a single 19.5 GB shard, so both the
download and the merge are minutes, not hours. The script resumes shard by shard, writes
`MERGE_REPORT.json` (per-tensor relative delta and changed-element fraction), copies
every other file byte for byte, and fails if any shard changed size.

No 800 GB box? `scripts/modal_app.py` runs download, merge and upload inside Modal
next to a persistent volume (this is how we shipped). Delete the base copy from the
volume afterwards; storage is billed monthly.

## 4. Verify before you publish (scripts/merge_check.py)

Serve the merged checkpoint (step 6), then score the same rendered validation rows on
both sides:

```
python scripts/merge_check.py render --dataset data/datasets/my-run --rows 16 --out check/val.jsonl
python scripts/merge_check.py tinker --rows check/val.jsonl --model tinker://<run-id>:train:0/sampler_weights/final --out check/tinker.jsonl
python scripts/merge_check.py tinker --rows check/val.jsonl --model thinkingmachines/Inkling-2-Mini --out check/base.jsonl
python scripts/merge_check.py sglang --rows check/val.jsonl --url http://<server>:30000 --out check/merged.jsonl
python scripts/merge_check.py compare check/tinker.jsonl check/merged.jsonl --rows check/val.jsonl
python scripts/merge_check.py compare check/base.jsonl check/merged.jsonl --rows check/val.jsonl
```

Ours, on 897K held-out tokens: merged NLL 0.647 vs Tinker adapter 0.644, mean |delta
logprob| 0.07 between them against 0.14 to the base. Then run `probe.py` against the
server with `--base-url`. A wrong merge (wrong scaling, a transposed delta, a skipped
shard) shows up here as a merged model that sits as close to the base as to the
adapter.

## 5. Upload

```
hf upload you/my-model ./merged . --exclude "MERGE_REPORT.json"
```

or `HfApi().upload_folder(...)` from Python (`upload_large_folder` was removed in
recent `huggingface_hub`). With `hf_xet` installed the Hub deduplicates chunks
against the base repository, so only the patched tensors are new. Start private, check the file list and sizes against
the base, then flip to public.

Model card essentials: base model and its license (Apache 2.0), what changed (which
tensors), the reasoning effort to run at, the system prompt shape and tool
expectations, the context window, hardware, and benchmark numbers with the harness
and the base measured on the same harness.

## 6. Serve

The merged checkpoint has the base's layout, so the base's recipes apply unchanged.

**LM Studio (preferred).** llama.cpp supports the `inkling2_loop` architecture, so
convert the merged checkpoint to GGUF with llama.cpp's `convert_hf_to_gguf.py` (or
quantize from the lmstudio-community GGUF lineage), or build an MLX 4-bit folder
(group size 64, affine). Drop the folder into the LM Studio models directory,
`lms load <model>`, and serve on `http://localhost:1234/v1`. Pair the GGUF with
the DSpark drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, from the unsloth GGUF repo)
for speculative decoding; the architecture also carries a built-in MTP head.
Verify with `probe.py --base-url http://localhost:1234/v1` and the `merge_check.py`
logprob comparison against the Tinker adapter before you trust the conversion —
quantization is lossy, so a GGUF/MLX build is a serving artifact, not the merge.

**SGLang** (pin a build with `inkling2_loop` support):

```
sglang serve --model-path you/my-model \
  --tp 2 --ep 2 --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 --page-size 128 \
  --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2 \
  --context-length 1048576 --host 0.0.0.0 --port 30000 \
  --speculative-algorithm DSPARK
```

The 19.5 GB checkpoint fits a single 80 GB GPU (`--tp 1`); scale TP/EP only for
concurrency.

**vLLM** (pin a build with `inkling2_loop` support):

```
vllm serve you/my-model --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Serving notes:

- DSpark speculative decoding proposes ~5 draft tokens per step and roughly
  doubles per-stream decode, but the draft shrinks the KV pool. Size concurrency
  to the pool that is left, or turn DSpark off for wide batches. (Measured on the
  parent skill's Inkling 2 Small run; the trade-off is architectural, the numbers
  were a 594B model's.)
- Warm the server with `GET /health` before a benchmark.

**PEFT adapter instead of a merge**: SGLang (`--lora-paths`) and vLLM
(`--enable-lora --lora-modules`) can apply the `-LoRA` repo on top of the base when
one deployment serves several fine-tunes. Check your serving version supports LoRA
for inkling2 first; the merged checkpoint has no such dependency.

**On Tinker**: the sampler path is a model name on the OpenAI- and
Anthropic-compatible endpoints (`references/model-facts.md`), up to 256K with a
transplant. No GPUs needed.
