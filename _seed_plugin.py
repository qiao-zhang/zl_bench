"""pytest plugin: seed random/numpy/torch RNG for reproducible benchmark inputs.

Data-dependent kernels (sort, topk, nonzero, ...) have value-dependent latency;
a fixed seed makes every run generate byte-identical inputs, so the latency
delta between two runs reflects the change under test, not the input data.
"""

from __future__ import annotations

import sys

_SEED = 0


def pytest_configure(config):
    seed = _SEED
    # random.seed too: some kits (cutlass_scaled_mm) pick cases via random.shuffle.
    import random
    random.seed(seed)
    seeded = ["random"]
    try:
        import numpy as _np
        _np.random.seed(seed)
        seeded.append("numpy")
    except Exception:
        pass
    try:
        import torch
    except Exception as exc:  # torch missing should never happen here, stay safe
        from _term_style import tag
        print(f"{tag('[seed-plugin]')} torch unavailable, "
              f"seeded {'+'.join(seeded)} only: {exc}",
              file=sys.stderr)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seeded.append("torch")
    from _term_style import tag
    print(f"{tag('[seed-plugin]')} manual_seed({seed}) for {'+'.join(seeded)} "
          "— reproducible benchmark inputs/cases",
          file=sys.stderr, flush=True)
