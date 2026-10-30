#!/usr/bin/env python3
"""Report tt.load/tt.store ops whose layout duplicates elements. Diagnostic only.

  python ir_layout_audit.py runs/ppu/key_multi_261029/gate        # a gate dir (base/ + new/)
  python ir_layout_audit.py runs/ppu/kda_fix_gate --quiet        # per-kernel summary only

Why this exists: RemoveLayoutConversions decides layouts by convert cost, and its
cost model prices a convert by tensor bytes only (getConvertUnitCost,
RemoveLayoutConversions.cpp:1133). A layout that hands every thread more elements
than the shape needs therefore looks free, even though it repeats the address/mask
math and the register footprint of the op producing the value.

How the two sides are compared — the details matter, because a total count hides a
regression that moves:

* a hit is keyed by (kernel file, **layout content**, shape, elems/thread,
  duplication), so a renamed alias matches (base `#linear` and new `#linear1` with
  the same body are the same layout) and a duplication change is never invisible;
* `worsened` reports a location whose largest duplication grew;
* `distribution changed` reports a location whose multiset of duplications differs
  without a larger hit — an 8x hit disappearing while a 2x hit grows to 4x is
  reported as **review**, never as "no new duplication".

The exit code is 0 by default: this is a report, not a gate — a diagnostic that
returns non-zero on the data it is asked to analyse cannot double as a smoke test.
Pass `--fail-on-regression` to get 1 when new adds a hit, a location worsens, or the
duplication distribution changes. The summary line names the case: `regression`,
`review` (distribution only), `reduced` (hits disappeared, nothing added) or
`no change` — only the last one means the two sides agree.

It is NOT a gate, and duplication is NOT memory traffic: the PPU lowering reuses
redundant registers (`refer back to the canonical load`,
third_party/ppu/lib/TritonPPUGPUToLLVM/LoadStoreOpToLLVM.cpp:285-302) and skips
redundant store elements (:505-514).

Scope: direct `#ttg.linear` layouts only (not blocked/slice/dot_op operand layouts);
static shapes of any rank; basis vectors are counted by dimension (a zero vector
still occupies an element, as in the Dialect's getTotalElemsPerThread); the
footprint is `2^register_basis_bits * threads / elements` and ignores the CTA
dimension.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

ALIAS_RE = re.compile(r"^(#[A-Za-z0-9_]+) = (#ttg\.linear<.*)$")
OP_RE = re.compile(r"\btt\.(load|store)\b")
TENSOR_RE = re.compile(r"tensor<([0-9]+(?:x[0-9]+)*)x[^,]+,\s*(#[A-Za-z0-9_]+)>")
NUM_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*(\d+)')
THREADS_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*(\d+)')
VEC_RE = re.compile(r"\[([^\[\]]*)\]")
INT_RE = re.compile(r"-?\d+")


class Hit(NamedTuple):
    """One load/store on a layout that assigns more elements than the shape needs."""

    file: str              # TTGIR path relative to the scanned side
    layout: str            # normalized layout body: the match key, alias-independent
    alias: str             # alias as printed, for the report only
    shape: str
    elems_per_thread: int
    duplication: float
    zero_lane_bases: int = 0


def bracket(text: str, start: int) -> str:
    """The [...] group that starts at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _strip_outer(group: str) -> str:
    """Drop the component's own brackets: `[]` has no vectors, `[[a], [b]]` has two."""
    inner = group.strip()
    if inner.startswith("[") and inner.endswith("]"):
        return inner[1:-1]
    return inner


def basis_vectors(group: str) -> int:
    """Number of basis vectors in a component such as `[[0, 1], [0, 32]]`.

    One vector is one basis bit, so this is the component's dimension, not the
    number of distinct offsets: a zero vector is a bit the layout leaves unused (it
    duplicates elements across registers/lanes) and the Dialect still hands those
    elements to every thread. Filtering zeros would underestimate the footprint —
    `[[0, 0], [0, 1]]` is 2 bits, not 1. Any rank works.
    """
    return len(VEC_RE.findall(_strip_outer(group)))


def zero_bases(group: str) -> int:
    """Basis vectors that are all zero, i.e. bits that do not move the offset."""
    return sum(1 for vec in VEC_RE.findall(_strip_outer(group))
               if not any(int(x) != 0 for x in INT_RE.findall(vec)))


def component(body: str, name: str) -> str:
    """The `[...]` group that follows `name =` in a layout body, or ""."""
    m = re.search(name + r"\s*=\s*\[", body)
    return bracket(body, m.end() - 1) if m else ""


def normalize(body: str) -> str:
    """Layout identity: its own text without whitespace, so an alias rename matches."""
    return re.sub(r"\s+", "", body)


def scan(path: Path, rel: str) -> Tuple[List[Hit], int]:
    """(hits, threads) for one TTGIR dump."""
    text = path.read_text(errors="replace")
    warps = NUM_WARPS_RE.search(text)
    lanes = THREADS_RE.search(text)
    threads = (int(warps.group(1)) if warps else 2) * (int(lanes.group(1)) if lanes else 32)

    layouts: Dict[str, Tuple[int, int, str]] = {}
    for line in text.splitlines():
        m = ALIAS_RE.match(line.strip())
        if m:
            body = m.group(2)
            reg, lane = component(body, "register"), component(body, "lane")
            layouts[m.group(1)] = (basis_vectors(reg) if reg else 0,
                                   zero_bases(lane) if lane else 0,
                                   normalize(body))

    hits: List[Hit] = []
    for line in text.splitlines():
        if not OP_RE.search(line):
            continue
        for shape, alias in TENSOR_RE.findall(line):
            if alias not in layouts:
                continue
            reg_bits, zero_lane, content = layouts[alias]
            elems = 1
            for dim in shape.split("x"):
                elems *= int(dim)
            per_thread = 1 << reg_bits
            dup = per_thread * threads / elems if elems else 0.0
            if dup > 1.0:
                hits.append(Hit(rel, content, alias, shape, per_thread, round(dup, 2),
                                zero_lane))
    return hits, threads


def side_dirs(root: Path) -> List[Tuple[str, Path]]:
    """(side, dir) pairs: a gate dir has base/ and new/, otherwise one side."""
    if (root / "base").is_dir() and (root / "new").is_dir():
        return [("base", root / "base"), ("new", root / "new")]
    return [(root.name, root)]


def collect(root: Path) -> Dict[str, List[Hit]]:
    """{side: hits} for one gate dir (or any dir with TTGIR dumps below it)."""
    out: Dict[str, List[Hit]] = {}
    for side, path in side_dirs(root):
        hits: List[Hit] = []
        for f in sorted(path.rglob("*.ttgir")):
            found, _ = scan(f, str(f.relative_to(path)))
            hits.extend(found)
        out[side] = hits
    return out


def _key(hit: Hit) -> Tuple[str, str, str, int, float]:
    """Match key: file + layout body + shape + elems/thread + duplication.

    Duplication is part of it on purpose: when it changes (a different `num-warps`,
    or the same footprint reading differently) an unchanged key would report
    "nothing happened".
    """
    return (hit.file, hit.layout, hit.shape, hit.elems_per_thread, hit.duplication)


def diff_sides(base: List[Hit], new: List[Hit]) -> Tuple[Counter, Counter, List[Tuple]]:
    """(added, removed, worsened) between the two sides.

    added/removed are per match key, so a hit that moves between kernels shows up as
    both instead of cancelling out. `worsened` compares the largest duplication each
    side reaches at the same (file, shape) — it cannot see a larger hit vanishing
    while a smaller one grows, which `distribution_changes` covers.
    """
    base_c, new_c = Counter(_key(h) for h in base), Counter(_key(h) for h in new)
    pos = lambda hits: {(h.file, h.shape): max(x.duplication for x in hits
                                              if (x.file, x.shape) == (h.file, h.shape))
                        for h in hits}
    base_pos, new_pos = pos(base), pos(new)
    worsened = [(p, base_pos[p], new_pos[p]) for p in sorted(base_pos.keys() & new_pos.keys())
                if new_pos[p] > base_pos[p]]
    return new_c - base_c, base_c - new_c, worsened


def distribution_changes(base: List[Hit], new: List[Hit]) -> List[Tuple]:
    """[(file, shape, base multiset, new multiset)] where the duplications differ.

    Catches what a max() comparison misses: an 8x hit disappearing while the 2x hit
    of the same kernel/shape grows to 4x leaves the max lower but the layout worse.
    """
    def by_pos(hits: List[Hit]) -> Dict[Tuple[str, str], Counter]:
        out: Dict[Tuple[str, str], Counter] = {}
        for h in hits:
            out.setdefault((h.file, h.shape), Counter())[h.duplication] += 1
        return out
    base_pos, new_pos = by_pos(base), by_pos(new)
    return [(f, s, base_pos[(f, s)], new_pos[(f, s)])
            for (f, s) in sorted(base_pos.keys() & new_pos.keys())
            if base_pos[(f, s)] != new_pos[(f, s)]]


def classify(added: Counter, removed: Counter, worsened: List[Tuple],
             changed: List[Tuple]) -> str:
    """One word for the summary line: regression / review / reduced / no change.

    `reduced` exists because a hit disappearing is a difference too — reporting it as
    "no change" would contradict the promise that only `no change` means the sides
    agree.
    """
    if added or worsened:
        return "regression"
    if changed:
        return "review"
    if removed:
        return "reduced"
    return "no change"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="gate dir (base/ + new/) or any dir with *.ttgir below it")
    ap.add_argument("--quiet", action="store_true", help="per-kernel summary only")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 when new adds a hit, a location worsens, or the "
                         "distribution changes (default: report only, exit 0)")
    args = ap.parse_args()

    exit_code = 0
    for root in args.dirs:
        sides = collect(root)
        for side, hits in sides.items():
            files = {h.file for h in hits}
            print(f"== {root.name}/{side}: {len(hits)} duplicating load/store hits "
                  f"in {len(files)} kernel(s)")
            if not args.quiet:
                for h in hits:
                    extra = f", {h.zero_lane_bases} zero lane basis" if h.zero_lane_bases else ""
                    print(f"   {h.file}: {h.alias} tensor<{h.shape}> "
                          f"{h.elems_per_thread} elems/thread -> {h.duplication}x{extra}")
        if not {"base", "new"} <= set(sides):
            continue

        added, removed, worsened = diff_sides(sides["base"], sides["new"])
        changed = distribution_changes(sides["base"], sides["new"])
        for (file, layout, shape, per_thread, dup), n in sorted(added.items()):
            print(f"   added in new: {file} {shape} {per_thread} elems/thread -> {dup}x "
                  f"(x{n}) [{layout[:60]}]")
        for (file, layout, shape, per_thread, dup), n in sorted(removed.items()):
            print(f"   dropped from base: {file} {shape} {per_thread} elems/thread -> "
                  f"{dup}x (x{n}) [{layout[:60]}]")
        for (file, shape), before, after in worsened:
            print(f"   worsened: {file} tensor<{shape}> {before}x -> {after}x")
        for file, shape, before, after in changed:
            fmt = lambda c: "{" + ", ".join(f"{k}x:{v}" for k, v in sorted(c.items())) + "}"
            print(f"   distribution changed: {file} tensor<{shape}> "
                  f"{fmt(before)} -> {fmt(after)}")

        kind = classify(added, removed, worsened, changed)
        if kind == "regression":
            print(f"== regression: {sum(added.values())} new hit(s), "
                  f"{len(worsened)} worsened location(s)")
            exit_code |= int(args.fail_on_regression)
        elif kind == "review":
            print("== review: duplication distribution changed, no larger hit — "
                  "not a clean bill, check these locations")
            exit_code |= int(args.fail_on_regression)
        elif kind == "reduced":
            print(f"== reduced: {sum(removed.values())} duplicating hit(s) gone from new, "
                  f"nothing added")
        else:
            print("== no change: the two sides report the same duplicating hits")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
