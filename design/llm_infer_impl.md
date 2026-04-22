# LLM Inference Framework Implementation Plan

## Goal

Build a serving framework that accepts a user prompt, runs:

1. prompt tokenization
2. model prefill
3. iterative decode
4. sampling / stop checking
5. token detokenization and streaming back to the user

The framework should also accept a user-provided local model directory and load
config, tokenizer assets, and weights from that directory through a pluggable
model-loader layer.

The framework should fit the current codebase, where:

- `model/qwen3_14b_prefill.py` should be treated as a prompt-stage kernel that
  consumes `[batch, seq, hidden]` and writes paged KV cache.
- `model/qwen3_14b_decode.py` implements a one-step decode kernel that consumes
  `[batch, hidden]`, `seq_lens`, `block_table`, and `slot_mapping`, which already
  implies a serving-oriented paged KV-cache interface.

## What Already Exists

The existing kernels suggest the correct serving split:

- Prefill is prompt ingestion and cache population.
- Decode is single-step token generation against an existing KV cache.
- Decode already exposes the runtime metadata needed by a scheduler:
  `block_table`, `slot_mapping`, and per-request `seq_lens`.

That means the missing work is not “another model kernel”. The missing work is the
runtime framework around those kernels.

## Target Architecture

Use a layered design:

1. API layer
2. Request/session layer
3. Scheduler
4. KV-cache manager
5. Executor
6. Model loader
7. Sampler
8. Streamer

### 1. API Layer

Responsibilities:

- Accept model initialization from a local model directory.
- Expose a simple `generate` interface.
- Support both non-streaming and streaming responses.
- Validate generation config.
- Convert user-visible requests into internal `InferenceRequest`.

Suggested interface:

```python
class GenerateConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] = []
    stream: bool = True


class InitModelRequest(BaseModel):
    model_id: str
    model_dir: str
    model_format: str | None = None


class GenerateRequest(BaseModel):
    request_id: str
    model_id: str
    prompt: str
    config: GenerateConfig = GenerateConfig()


class LLMEngine:
    def init_model(self, model_id: str, model_dir: str) -> None: ...
    def generate(self, model_id: str, prompt: str, config: GenerateConfig) -> str | Iterator[str]: ...
```

Suggested internal composition:

```python
class LLMEngine:
    def __init__(
        self,
        model_registry: "ModelRegistry",
        kv_cache_manager: KvCacheManager,
        executor: ModelExecutor,
        sampler: Sampler,
    ) -> None: ...
```

`generate()` should hide the full inference pipeline:

1. tokenize the input prompt
2. lookup token embeddings
3. allocate prompt KV pages through `KvCacheManager`
4. run multi-layer prefill
5. run decode loop with manager-provided `block_table` and `slot_mapping`
6. sample next token
7. detokenize generated tokens
8. free KV pages when the request finishes or aborts
9. return or stream text

### 2. Request / Session Layer

Each active request needs stable state across prefill and decode.

Suggested request state:

```python
@dataclass
class RequestState:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    sampling_params: SamplingParams
    status: Literal["waiting", "prefill", "decode", "finished", "aborted", "error"]
    max_new_tokens: int
    stop_strings: list[str]
    eos_token_id: int
    seq_len: int
    num_prompt_tokens: int
    kv_allocation: "KvAllocation | None"
    last_hidden: torch.Tensor | None
    output_text: str
```

Important design point:

- `prefill` should produce the KV state for the prompt and the hidden state for the
  final prompt token.
- `decode` should consume the embedding/hidden state of the most recent token and
  append one more token to the same request state.

### Model Config Data Structures

The user-provided model directory needs to be converted into stable
internal config objects. These should be stored once and referenced by `model_id`
during generation.

Suggested config types:

```python
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    torch_dtype: str


@dataclass(frozen=True)
class RuntimeConfig:
    page_size: int
    max_batch_size: int
    max_seq_len: int
    device: str
    kv_dtype: str
    weight_dtype: str


@dataclass(frozen=True)
class LayerSpec:
    layer_idx: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass
class ModelRecord:
    config: ModelConfig
    runtime: RuntimeConfig
    tokenizer: "TokenizerAdapter"
    layer_specs: list[LayerSpec]
```

Design rule:

- `ModelConfig` mirrors validated model architecture from the source format.
- `RuntimeConfig` stores serving/runtime choices.
- `LayerSpec` is the per-layer shape contract used to instantiate the single-layer
  kernels.
- `ModelRecord` is what the engine registers and looks up during `generate()`.

### 3. Scheduler

The scheduler is the core of the framework.

It should maintain three queues:

- `waiting_prefill`
- `ready_decode`
- `finished`

Its job is to form efficient batches under memory and latency constraints.

Two scheduling loops are enough for an MVP:

- Prefill loop:
  collect new requests, pad/pack them into a prefill batch, allocate KV pages,
  run prefill once.
- Decode loop:
  repeatedly batch active requests for one decode step per request.

Recommended policy:

- Give decode higher priority than prefill to reduce token latency.
- Reserve a small fraction of capacity for prefill so new requests do not starve.
- Use continuous batching:
  when one request finishes, immediately admit a waiting request without draining
  the whole decode batch.

### 4. KV-Cache Manager

This is a required component because both `prefill` and `decode` operate on the
same paged KV-cache abstraction.

Responsibilities:

- Own all KV memory.
- Allocate logical pages per request.
- Maintain logical-to-physical page mapping (`block_table`).
- Compute `slot_mapping` for the next generated token.
- Reclaim pages when requests finish or abort.

Suggested abstractions:

```python
@dataclass
class KvAllocation:
    request_id: str
    block_ids: list[int]
    tokens_capacity: int
    tokens_used: int


class KvCacheManager:
    def allocate_for_prompt(self, request_id: str, prompt_len: int) -> KvAllocation: ...
    def ensure_one_more_slot(self, alloc: KvAllocation) -> int: ...
    def block_table_for_batch(self, requests: list[RequestState]) -> torch.Tensor: ...
    def slot_mapping_for_batch(self, requests: list[RequestState]) -> torch.Tensor: ...
    def free(self, alloc: KvAllocation) -> None: ...
```

The design requirement is:

- KV cache must be managed only through `KvCacheManager`.
- `generate()` must not manipulate raw page tables or cache tensors directly.
- executor reads cache metadata from the manager and writes K/V tensors into the
  slots it assigns.

Suggested responsibilities split:

- `KvCacheManager`: owns capacity, allocation, reuse, block tables, slot mapping,
  and lifecycle
- `ModelExecutor`: consumes manager-provided metadata and runs kernels
- `Scheduler`: decides which requests run next, but does not own cache state

### Key Design Decision: Shared Paged KV Layout

Assume `prefill` and `decode` both use the same paged KV-cache layout.

Reason:

- `decode` already depends on paged metadata.
- Paged cache is the correct design for long-running multi-request serving.
- It avoids compaction when requests finish at different times.

That means:

- the KV manager is the single source of truth for page ownership
- prefill allocates and fills prompt pages directly
- decode appends into the same allocation without any layout conversion
- `block_table` and `slot_mapping` are framework-level metadata shared by both
  stages

### 5. Executor

The executor isolates model-runtime details from scheduling logic.

Suggested responsibilities:

- load model weights
- initialize runtime handles
- compile/load prefill and decode programs
- build input tensors
- launch kernels
- return outputs in framework-native format

Because the provided kernels are single-layer kernels, the executor must assemble
the full model by repeating the layer computation according to
`config.num_hidden_layers`.

Suggested interface:

```python
class ModelExecutor:
    def run_prefill(self, batch: "PrefillBatch") -> "PrefillResult": ...
    def run_decode(self, batch: "DecodeBatch") -> "DecodeResult": ...
```

Suggested internal structure:

```python
@dataclass
class LayerWeights:
    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class RuntimeModel:
    config: ModelConfig
    runtime: RuntimeConfig
    embed_tokens: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head: torch.Tensor
    layers: list[LayerWeights]
```

Execution rule:

- input token ids are first converted to hidden states through `embed_tokens`
- prefill runs layer `0..num_hidden_layers-1` sequentially
- decode runs layer `0..num_hidden_layers-1` sequentially for each step
- after the last layer, apply final norm and `lm_head` to produce logits

Suggested batch/result shapes:

```python
@dataclass
class PrefillBatch:
    request_ids: list[str]
    token_ids: torch.Tensor
    input_embeddings: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    kv_allocations: list[KvAllocation]


@dataclass
class PrefillResult:
    last_hidden: torch.Tensor
    logits: torch.Tensor


@dataclass
class DecodeBatch:
    request_ids: list[str]
    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class DecodeResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor
```

Note:

- Your current kernels return hidden/output activations, not final vocabulary logits.
- Real serving still needs an LM-head projection and sampler after that.
- If the LM head is not fused yet, keep it as a runtime-side matmul.

### 6. Model Loader

The framework needs a model-loader layer because the user may provide model
artifacts in different local directory formats rather than a single fixed format.

Responsibilities:

- validate the directory structure
- load tokenizer files
- parse `config.json`
- locate weight shards such as `*.safetensors`
- map source-format parameter names into the tensors expected by the executor
- build validated `ModelConfig`
- build `LayerSpec` entries for all layers
- load per-layer weights and shared weights:
  - token embedding
  - final norm
  - lm head
  - layer weights for `0..num_hidden_layers-1`

Suggested interface:

```python
@dataclass
class ModelRuntimeMeta:
    architecture: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int


@dataclass
class LoadedModel:
    model_id: str
    model_dir: str
    config: ModelConfig
    tokenizer: "TokenizerAdapter"
    runtime_model: RuntimeModel


class ModelLoader:
    def load(self, model_id: str, model_dir: str, model_format: str | None = None) -> LoadedModel: ...
```

Expected files in `model_dir` depend on the selected or detected format.

For the current Hugging Face adapter, expected files are:

- `config.json`
- tokenizer files such as `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`
- weight files such as `model.safetensors` or sharded `*.safetensors`

Important design point:

- Treat source-format naming as an external format.
- Convert it once at load time into internal executor-owned tensors.
- Do not let scheduler or serving code depend on source-format parameter names.
- Build the full model shape from `num_hidden_layers`, `hidden_size`,
  `num_attention_heads`, `num_key_value_heads`, and `intermediate_size` in the
  validated config, not from hard-coded constants.

### 7. Sampler

The sampler converts logits into the next token.

Responsibilities:

- temperature scaling
- top-k / top-p filtering
- greedy fallback
- repetition penalty if needed later
- EOS detection

Suggested interface:

```python
class Sampler:
    def sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor: ...
```

For the first implementation:

- support greedy and temperature + top-p
- keep beam search out of scope

### 8. Streamer

Streaming should be event-driven.

Each request emits events:

- `request_started`
- `token_generated`
- `request_finished`
- `request_error`

The API layer can convert these events into:

- SSE
- websocket messages
- buffered final JSON

## End-to-End Execution Flow

### 0. Model Initialization

1. user provides `model_dir`
2. model loader detects or validates the requested model format
3. tokenizer is loaded from the directory
4. config is parsed into `ModelConfig`
5. layer specs are built for `num_hidden_layers`
6. weights are loaded and converted into `RuntimeModel`
7. executor is initialized with that loaded model
8. the model is registered under `model_id`

### A. New Request

1. receive prompt
2. resolve target model
3. tokenize prompt into `prompt_token_ids`
4. lookup token embeddings
5. allocate prompt pages through `KvCacheManager`
6. create `RequestState`
7. enqueue into `waiting_prefill`

### B. Prefill Stage

1. scheduler picks several waiting requests
2. `KvCacheManager` validates and finalizes prompt allocation state
3. executor builds the prefill batch from prompt embeddings and manager metadata
4. executor runs all transformer layers sequentially in prefill mode
5. framework stores:
   - updated KV cache
   - last-token hidden state
   - optionally logits for first sampled token
6. sampler picks the first generated token
7. request moves to `ready_decode`

### C. Decode Stage

For each decode iteration:

1. scheduler forms a decode batch from active requests
2. `KvCacheManager` grows allocations if another token slot is needed
3. `KvCacheManager` computes `block_table` and `slot_mapping`
4. executor performs embedding lookup for the latest sampled token
5. executor runs all transformer layers sequentially in decode mode
6. LM head produces logits
7. sampler selects one token per request
8. framework appends token to each request
9. stop conditions are checked:
   - EOS reached
   - stop string matched
   - `max_new_tokens` reached
10. finished requests are removed and KV pages are freed by `KvCacheManager`
11. unfinished requests stay in `ready_decode`

### D. Response Stage

- In streaming mode, send each token as soon as sampled.
- In non-streaming mode, accumulate final text and return at completion.

## Recommended Repo Structure

```text
design/
  llm_infer_impl.md
llm/
  core/
    api.py
    server.py
    model_loader.py
    scheduler.py
    request_state.py
    kv_cache.py
    executor.py
    sampler.py
    streamer.py
    tokenizer.py
    types.py
model/
  qwen3_14b_prefill.py
  qwen3_14b_decode.py
tests/
  test_scheduler.py
  test_kv_cache.py
  test_sampler.py
  test_end_to_end.py
```

## Concrete Implementation Plan

### Phase 1: Define Stable Interfaces

Deliverables:

- `RequestState`, `SamplingParams`, batch/result dataclasses
- `ModelConfig`, `RuntimeConfig`, `LayerSpec`, `ModelRecord`
- `LoadedModel`, tokenizer adapter
- executor interface
- KV manager interface
- scheduler interface

Why first:

- Right now the kernels exist, but there is no stable contract between runtime,
  memory manager, and serving loop.

### Phase 2: Build a Single-Request Golden Path

Deliverables:

- initialize model from a local model directory
- one request in, one full answer out
- no streaming yet
- greedy sampling only
- fixed batch size of 1

Flow:

1. load local model directory
2. tokenize prompt
3. lookup embeddings
4. run multi-layer prefill
5. run final norm + LM head
6. sample next token
7. loop multi-layer decode until stop
8. detokenize and return output

Why second:

- This is the shortest path to verify that prefill/decode handoff is correct.

### Phase 3: Add KV-Cache Manager and Paged Decode State

Deliverables:

- paged cache allocation/free
- `block_table` generation
- `slot_mapping` generation
- capacity checks

Validation:

- multi-request decode with different context lengths
- correct freeing and reuse of blocks

### Phase 4: Add Continuous Batching

Deliverables:

- prefill queue
- decode queue
- batch formation policy
- admission control

Metrics to track:

- TTFT: time to first token
- TPOT: time per output token
- active requests
- cache occupancy

### Phase 5: Add Streaming API

Deliverables:

- SSE or websocket streaming
- cancellation support
- per-request error propagation

### Phase 6: Optimize the Prefill/Decode Boundary

This is now mostly about batch construction and metadata plumbing rather than
cache layout conversion.

Target:

- make prefill and decode consume the same request metadata model
- reuse page allocations across the prefill-to-decode transition with no copy
- minimize scheduler overhead when a request moves from prompt stage to decode

### Phase 7: Reliability and Ops

Deliverables:

- structured logging
- per-stage timing
- request tracing
- graceful shutdown
- health/readiness endpoints

## Minimal MVP

If the goal is to get something working quickly, build only this first:

1. single model
2. single process
3. single device
4. local Hugging Face model directory input
5. greedy decode
6. batch size 1 for prefill
7. continuous decode batching optional
8. paged KV manager
9. streaming text output

This is enough to validate the framework shape before adding throughput
optimizations.

## Main Risks

### 1. Prefill/Decode Metadata Contract Drift

With shared paged KV layout, the main risk moves from memory layout mismatch to
metadata mismatch.

- Prefill and decode must agree on page size, logical-block indexing, and slot
  ownership semantics.
- If `seq_lens`, `block_table`, and `slot_mapping` are interpreted differently in
  the two stages, the serving loop will still fail even though the cache layout is
  unified.

### 2. Missing LM Head in the Serving Contract

Your kernels produce hidden/output states, but generation requires logits.

You need a clear rule for where vocabulary projection lives:

- fused into kernels later, or
- runtime-side projection for now

### 3. Hugging Face Weight Mapping Errors

The user may provide an arbitrary local model directory.

Common failure modes:

- unsupported architecture in `config.json`
- missing tokenizer files
- weight names that do not match the expected Qwen mapping
- shard loading or dtype conversion mistakes

The model loader should fail fast with explicit validation errors.

### 4. Single-Layer Kernel / Multi-Layer Model Assembly

The provided kernels are only one transformer layer.

The executor therefore has to assemble the full model by:

- building layer specs from `num_hidden_layers`
- loading one `LayerWeights` object per layer
- looping prefill/decode through all layers in order

If the layer order, residual contract, or final norm / lm-head placement is
wrong, generation will be incorrect even if each layer kernel is correct.

### 5. Scheduler / Memory Coupling

Batching decisions depend on KV capacity. The scheduler and KV manager cannot be
designed independently.

### 6. Stop-String Handling

Token-level EOS is easy. Stop strings are not.

Do not check stop strings only on whole-text finalization. They must be checked
incrementally during streaming.

## Recommended Next Coding Steps

1. Create framework dataclasses and interfaces.
2. Implement a pluggable `ModelLoader` with format adapters.
3. Implement a minimal `KvCacheManager`.
4. Implement `ModelExecutor` wrappers around the existing prefill/decode programs.
5. Build a single-request generation loop.
6. Add a tiny API surface for streaming output.
7. Then harden the shared prefill/decode metadata contract and batch policies.

## Opinionated Recommendation

Do not start from the HTTP server first.

The correct order is:

1. model loader + kernel contract
2. KV cache manager
3. single-request generation loop
4. multi-request scheduler
5. streaming API

If you reverse that order, you will end up rewriting the serving layer once the
prefill/decode state contract becomes real.
