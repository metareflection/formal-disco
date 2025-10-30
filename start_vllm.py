#!/usr/bin/env python3
"""
Script to start a vLLM server with OpenAI-compatible API.

Usage:
    python vllm.py --model Qwen/Qwen3-4B-Instruct-2507 --port 8000

Environment variables:
    VLLM_MODEL: Model to serve (default: Qwen/Qwen3-4B-Instruct-2507)
    VLLM_HOST: Host to bind to (default: 0.0.0.0)
    VLLM_PORT: Port to bind to (default: 8000)
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible server")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        help="Model to serve (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("VLLM_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VLLM_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="Number of GPUs to use for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.9")),
        help="GPU memory utilization (default: 0.9)",
    )

    args = parser.parse_args()

    # Import vllm here to provide better error messages
    try:
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        print("Error: vLLM is not installed. Please install it with:")
        print("  pip install vllm")
        sys.exit(1)

    print(f"Starting vLLM server with model: {args.model}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1")

    # Configure engine arguments
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Start the server
    run_server(
        engine_args,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
