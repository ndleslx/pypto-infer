from __future__ import annotations

import argparse
from pathlib import Path

from llm.core import GenerateConfig, LLMEngine, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU-only Qwen3-14B generation with the reference executor."
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory, e.g. a Hugging Face snapshot.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default="qwen3-14b-cpu-ref")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    engine = LLMEngine()
    engine.init_model(
        model_id=args.model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=64,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
    )
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
    )
    if args.stream:
        result = engine.generate(args.model_id, args.prompt, config)
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        result = engine.generate_result(args.model_id, args.prompt, config)
        print(f"text: {result.text}")
        print(f"token_ids: {result.token_ids}")
        print(f"finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    main()
