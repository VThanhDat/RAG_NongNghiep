"""
Benchmark the configured embedding model.

Usage:
    python -B embedding/benchmarks/benchmark_embedding.py --repeat 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding import get_embedder_from_config


SAMPLE_TEXTS = [
    "Benh than thu tren sau rieng thuong phat sinh khi am do cao.",
    "Can tia canh tao thong thoang va ve sinh vuon de han che nam benh.",
    "Bang lieu luong:\n| Cay trong | Phan bon |\n| --- | --- |\n| Lua | Dam, lan, kali |",
    "Dat trong ca phe can thoat nuoc tot va bo sung huu co dinh ky.",
    "Sau hai tren rau mau can duoc theo doi som de phong tru kip thoi.",
] * 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pipeline = get_embedder_from_config(cfg)
    texts = SAMPLE_TEXTS * max(1, args.repeat)

    start = time.perf_counter()
    result = pipeline.embed_documents(texts)
    duration = time.perf_counter() - start

    dense = result["dense"]
    dim = len(dense[0]) if dense else 0
    per_chunk = duration / max(1, len(texts))
    throughput = len(texts) / duration if duration else 0.0

    print(f"provider={pipeline.dense_provider}")
    print(f"model={pipeline.dense_model}")
    print(f"chunks={len(texts)}")
    print(f"dimension={dim}")
    print(f"seconds={duration:.3f}")
    print(f"seconds_per_chunk={per_chunk:.4f}")
    print(f"chunks_per_second={throughput:.2f}")


if __name__ == "__main__":
    main()
