# pypto-infer

This repository is a minimal LLM inference framework built around a simple `generate` interface and PyPTO-backed model execution. The current implementation is aimed at local, single-request inference for Qwen3-14B using bundled single-layer kernels under `model/`, with the model weights loaded from a user-provided local directory.

## What This Repo Implements

The current pipeline is:

1. Tokenize the prompt.
2. Lookup token embeddings.
3. Run model prefill.
4. Run the decode loop.
5. Sample the next token.
6. Detokenize generated tokens.
7. Return text and generated token IDs.

The framework is designed so the frontend API stays simple while the backend can be swapped between a reference executor and a device executor.

## Repository Layout

- `llm/core/`
  Core runtime code.
- `llm/core/engine.py`
  High-level entry point. Exposes model initialization and generation.
- `llm/core/model_loader.py`
  General model loading abstraction. The current concrete adapter loads a local Hugging Face-style directory, but the loader interface is not restricted to Hugging Face.
- `llm/core/types.py`
  Shared config and runtime data structures.
- `llm/core/kv_cache.py`
  Paged KV cache manager.
- `llm/core/executor.py`
  Reference PyTorch executor.
- `llm/core/pypto_executor.py`
  PyPTO-backed executor that composes the full model from the provided single-layer kernels.
- `model/`
  Kernel implementations. Prefill was extended to accept paged-KV inputs.
- `examples/`
  Example entry points. `qwen3_14b_local_generate.py` is the main smoke-test example.
- `design/`
  Notes on the implementation direction and architecture.

## Core Design

### 1. Simple API

The main external interface is `LLMEngine`:

```python
engine.init_model(...)
text = engine.generate(model_id, prompt)
result = engine.generate_result(model_id, prompt)
```

`generate_result()` is the better interface for debugging because it returns:

- generated text
- generated token IDs
- finish reason

### 2. General Model Loader

The loader is intentionally abstracted so that the repo can later support:

- local Hugging Face snapshots
- custom weight layouts
- converted offline formats
- future model registries

At the moment, the implemented path expects a local directory with config, tokenizer files, and weight shards for Qwen3-14B.

### 3. Model Config and Runtime Config

Two distinct config layers are used:

- `ModelConfig`
  Static model shape and tokenizer-related metadata.
- `RuntimeConfig`
  Execution-time controls such as page size, max sequence length, KV dtype, and device placement.

This separation is important because the user-provided model directory defines the model shape, while the runtime environment defines how the model is executed.

### 4. Single-Layer Kernel Composition

The bundled kernels only implement one transformer layer, so the framework builds the full model by:

- loading all layer weights from the model directory
- iterating through every layer in prefill
- iterating through every layer again during decode
- maintaining KV state across the full request

The current PyPTO executor validates that the loaded model matches the Qwen3-14B layer shape expected by the bundled kernels.

### 5. Paged KV Cache

KV cache is managed centrally in `KvCacheManager`.

Responsibilities:

- allocate pages for a request
- extend capacity during decode
- map logical token positions to physical KV slots
- materialize cache views for device kernels
- free pages when the request finishes

The prefill kernel was updated to accept a paged `block_table` contract so prefill and decode now use the same paging model.

## How To Run

### Local Python Compile Check

```bash
python -m compileall llm examples
```

### NPU Smoke Test

The user requested that device execution use `task-submit`. The example below follows that contract:

```bash
task-submit --device auto --run "python examples/qwen3_14b_local_generate.py --model-dir /data/linyifan/models/Qwen3-14B --prompt 'Hello' --platform a2a3 --pypto-root /data/liuxu/pypto --max-new-tokens 1"
```

The example prints:

- `text`
- `token_ids`
- `finish_reason`

## Current Status

What is working:

- model loading from a local directory
- tokenizer integration
- full prefill and decode control flow
- paged KV cache management
- PyPTO kernel compilation and execution
- NPU end-to-end smoke execution through `task-submit`

What is not finished:

- numerical correctness of the hardware path

The current hardware run can complete end-to-end, but the logits are not yet numerically stable. The sampler currently sanitizes non-finite logits to prevent crashes during smoke tests. That makes the control flow testable, but the generated tokens should not yet be treated as correct model output.

## Recommended Next Steps

1. Add per-layer validation between `ModelExecutor` and `PyptoQwen14BExecutor`.
2. Compare prefill outputs layer by layer and identify the first divergence.
3. Do the same for decode with a fixed prompt and fixed token input.
4. Verify weight layout, transpose direction, and RMSNorm handling against the kernel contract.
5. Once logits are finite and aligned, remove the sampler fallback from the critical path.

## Developer Notes

- Keep new code under `llm/`.
- Keep kernel code under `model/`.
- Put runnable demos in `examples/`.
- Use `task-submit --device auto --run "python ..."` for NPU execution.
- Prefer `generate_result()` over `generate()` when debugging token-level behavior.

