"""Shared timing helpers for intake benchmarks.

Noise is the full sample spread: (max - min) / median * 100.
Timed regions run with the garbage collector disabled, after warmup.
"""

from __future__ import annotations

import gc
import json
import time


def sync(value) -> None:
    import jax

    for leaf in jax.tree.leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def measure(fn, *, repeats: int = 7, warmup: int = 2, sync_output: bool = False) -> dict:
    if repeats < 5:
        raise ValueError("intake baselines need at least 5 samples")
    last = None
    for _ in range(warmup):
        last = fn()
        if sync_output:
            sync(last)
    gc.collect()
    gc.disable()
    samples = []
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            last = fn()
            if sync_output:
                sync(last)
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        gc.enable()
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (ordered[-1] - ordered[0]) / median * 100.0
    return {
        "median_ms": median,
        "noise_pct": noise,
        "samples_ms": samples,
        "result": last,
    }


def emit(name: str, stats: dict, checksum: float) -> None:
    payload = {
        "benchmark": name,
        "median_ms": round(stats["median_ms"], 6),
        "noise_pct": round(stats["noise_pct"], 6),
        "samples_ms": [round(sample, 6) for sample in stats["samples_ms"]],
        "checksum": checksum,
    }
    print(json.dumps(payload, sort_keys=True))
