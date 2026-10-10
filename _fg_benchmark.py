"""Shared access to the FlagGems benchmark framework under test.

Both upstream repos ship their own `benchmark` package (benchmark/base.py with
a `Benchmark` root class); whichever repo pytest is running resolves here, so
the plugins never import flag_gems or flaggems_vllm by name.
"""

from __future__ import annotations

from typing import List


def benchmark_classes() -> List[type]:
    """`benchmark.base.Benchmark` and every subclass defined so far, each once.

    Call after collection: subclasses live in the test modules, so the list
    is only complete once they are imported.
    """
    from benchmark import base

    found, seen, stack = [], set(), [base.Benchmark]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue  # diamond inheritance
        seen.add(cls)
        found.append(cls)
        stack.extend(cls.__subclasses__())
    return found
