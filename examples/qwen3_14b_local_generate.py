from __future__ import annotations

import argparse
from pathlib import Path

from llm.core import GenerateConfig, LLMEngine, RuntimeConfig
from llm.core.kv_cache import KvCacheManager
from llm.core.pypto_executor import PyptoQwen14BExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Qwen3-14B generation with the bundled PyPTO kernels.")
    parser.add_argument("--model-dir", required=True, help="Local model directory, e.g. a Hugging Face snapshot.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default="qwen3-14b-local")
    parser.add_argument("--platform", default="a2a3sim", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pypto-root", default="/data/liuxu/pypto", help="Path to the local PyPTO repository.")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--save-kernels-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        pypto_root=args.pypto_root,
        platform=args.platform,
        device_id=args.device_id,
        save_kernels_dir=args.save_kernels_dir,
    )
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )
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
