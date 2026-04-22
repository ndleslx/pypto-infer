# Repository Guidelines

## Project Structure & Module Organization

- `llm/`: main source tree.
- `llm/core/`: inference engine, model loader, KV cache manager, sampler, executors, and shared types.
- `model/`: PyPTO kernel implementations. These are single-layer kernels composed into a full model by the executor.
- `examples/`: runnable examples, including `qwen3_14b_local_generate.py`.
- `design/`: implementation notes and architecture drafts.
- `README.md`: developer-facing overview and current project status.

There is no dedicated `tests/` directory yet. Validation is currently done through compile checks and example smoke runs.

## Build, Test, and Development Commands

- `python -m compileall llm examples`
  Verifies Python modules under `llm/` and `examples/` compile cleanly.
- `python examples/qwen3_14b_local_generate.py --model-dir <local-model-dir> --prompt "Hello"`
  Runs the example locally on CPU-hosted control flow.
- `task-submit --device auto --run "python examples/qwen3_14b_local_generate.py --model-dir /data/linyifan/models/Qwen3-14B --prompt 'Hello' --platform a2a3 --pypto-root /data/liuxu/pypto --max-new-tokens 1"`
  Runs the NPU smoke test using the required device submission flow.

## Coding Style & Naming Conventions

- Use Python with 4-space indentation and standard PEP 8 naming.
- Prefer `snake_case` for functions, variables, and module names; use `CamelCase` for classes.
- Keep new framework code under `llm/` and kernel code under `model/`.
- Keep APIs simple and explicit. Prefer structured dataclasses in `llm/core/types.py` for shared contracts.
- Use ASCII unless a file already requires non-ASCII content.

## Testing Guidelines

- Add compile checks for every change: `python -m compileall llm examples`.
- Prefer smoke tests through `examples/` until a formal test suite exists.
- When debugging generation, use `generate_result()` so token IDs and finish reasons are visible.
- If you add tests later, place them under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines

- Follow concise, imperative commit messages, for example: `Add PyPTO LLM inference scaffold`.
- Keep each commit scoped to one logical change.
- PRs should include:
  - a short summary of the user-visible or developer-visible change
  - exact commands used for validation
  - any model, device, or PyPTO assumptions
  - logs or sample output for NPU-related changes when relevant

## Architecture Notes

- The current implementation supports local model loading plus a paged-KV inference flow.
- The hardware path runs end-to-end, but numerical correctness is not finished yet. Treat generated output as a smoke-test signal unless validated against the reference executor.
