#!/usr/bin/env python3
"""Compare the TTGIR of every RLC sub-switch variant, side by side.

Input is one or more matrices produced by gen_combination_ttgir.sh, in either layout:

    <root>/<variant>/ttgir/<shape>/<kernel>/<file>.ttgir          one op
    <root>/<op>/<variant>/ttgir/<shape>/<kernel>/<file>.ttgir     several ops

A flat root is read as a single op named after the directory, so both layouts print
the same way when only one op is in play. `--op` picks ops out of a multi-op root,
`--dtype` keeps the files whose name carries one of the named dtypes (the dump writes
`..._w8s3__fp16.ttgir`), and `--kernel` keeps paths containing a substring.

What it prints, per (op), (shape, kernel, dtype):

  1. a matrix: `convert_layout` count per variant
  2. a layout fingerprint per variant, grouped so you can see which variants are
     *equivalent* (same set of encodings) without reading 10 files
  3. `diff` lines against the reference variant, using a normalised diff
     (loc attributes and SSA numbering stripped), so a "no change" really shows 0
  4. marginal attribution along the dependency chain
     (enhance_off -> c1b0s0r0 -> c1b1s0r0 -> c1b1s1r0 -> c1b1s1r1),
     i.e. "which switch introduced this change"

Usage:
    python compare_ttgir_matrix.py runs/ppu/rlc_matrix
    python compare_ttgir_matrix.py runs/ppu/rlc_matrix/coalesce           # one op's dir
    python compare_ttgir_matrix.py runs/ppu/rlc_matrix --op coalesce --dtype fp16
    python compare_ttgir_matrix.py runs/ppu/rlc_matrix --op coalesce,reduce --kernel _reduce
    python compare_ttgir_matrix.py runs/ppu/rlc_matrix --base c0b0s0r0 --csv matrix.csv

Exit status: 0 always (a difference is data); 2 if no ttgir was found.
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

RE_LAYOUT_DEF = re.compile(r"^#(\w+) = #ttg\.blocked<\{(.*)\}>$", re.M)
RE_CONVERT = re.compile(r"ttg\.convert_layout")
RE_LOC_ATTR = re.compile(r"\s*loc\([^)]*\)")
RE_SSA = re.compile(r"%([A-Za-z_][\w]*?)_\d+")
RE_NUM = re.compile(r"%(\d+)")

CHAIN = ["enhance_off", "c0b0s0r0", "c1b0s0r0", "c1b1s0r0", "c1b1s1r0", "c1b1s1r1"]
SWITCH_OF_STEP = {
    ("enhance_off", "c0b0s0r0"): "master(=1) + cost_based off",
    ("c0b0s0r0", "c1b0s0r0"): "cost_based 0->1",
    ("c1b0s0r0", "c1b1s0r0"): "backward_propagation 0->1",
    ("c1b1s0r0", "c1b1s1r0"): "small_component_solving 0->1",
    ("c1b1s1r0", "c1b1s1r1"): "store_layout_remat 0->1",
}

# `--dtype fp16` has to match the dump's `__fp16` suffix, and users think in either
# spelling (benchmark dtype vs tensor dtype), so each name expands to its aliases.
DTYPE_ALIASES = {
    "fp16": ("fp16", "float16", "half"),
    "bf16": ("bf16", "bfloat16"),
    "fp32": ("fp32", "float32"),
    "fp64": ("fp64", "float64", "double"),
}


def layout_labels(text):
    """`#blocked = ...{sizePerThread = [4], ...}` -> 'b[spt|tpw|warps|order]'."""
    out = {}
    for match in RE_LAYOUT_DEF.finditer(text):
        name, body = match.group(1), match.group(2)
        fields = {}
        for key in ("sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"):
            found = re.search(key + r" = (\[[^\]]*\])", body)
            fields[key] = found.group(1).replace(" ", "") if found else "?"
        out["#" + name] = "b[%s|%s|%s|%s]" % (
            fields["sizePerThread"], fields["threadsPerWarp"],
            fields["warpsPerCTA"], fields["order"])
    return out


def normalised_lines(text):
    """Strip loc attributes, loc defs and SSA numbering -> comparable lines."""
    lines = []
    for line in text.splitlines():
        if line.startswith("#loc"):
            continue
        line = RE_LOC_ATTR.sub("", line)
        line = RE_SSA.sub(r"%\1", line)
        line = RE_NUM.sub("%N", line)
        line = re.sub(r"#(\w+?)\d+ =", r"#\1 =", line)
        lines.append(line.strip())
    return lines


def analyse(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    labels = layout_labels(text)
    # Count how many IR lines mention each layout, by name boundary (`#blocked`
    # must not match `#blocked1`), skipping the definition lines themselves.
    used = Counter()
    for line in text.splitlines():
        if line.startswith("#") and "= #ttg." in line:
            continue
        for name, label in labels.items():
            if re.search(re.escape(name) + r"\b", line):
                used[label] += 1
    fingerprint = ",".join("%s x%d" % (k, v) for k, v in sorted(used.items()) if v)
    return {
        "converts": len(RE_CONVERT.findall(text)),
        "labels": labels,
        "fingerprint": fingerprint or "-",
        "lines": normalised_lines(text),
    }


def split_names(values):
    """`--op a --op b,c` (or one `--dtype`) -> {'a', 'b', 'c'}."""
    return {part.strip() for value in values or () for part in value.split(",") if part.strip()}


def dtype_tokens(values):
    """The dtype names to match against a filename, aliases expanded."""
    names = split_names(values)
    tokens = set()
    for name in names:
        tokens.update(DTYPE_ALIASES.get(name.lower(), (name.lower(),)))
    return tokens


def dtype_matches(rel, tokens):
    """A dump names its dtype in the file name (`..._w8s3__fp16.ttgir`)."""
    if not tokens:
        return True
    low = rel.lower()
    return any(token in low for token in tokens)


def matrices(root):
    """[(op, variant, ttgir_dir)] for both layouts under `root`."""
    found = []
    if not os.path.isdir(root):
        return found
    flat_op = os.path.basename(os.path.normpath(root))
    for name in sorted(os.listdir(root)):
        top = os.path.join(root, name)
        if not os.path.isdir(top):
            continue
        if os.path.isdir(os.path.join(top, "ttgir")):       # <root>/<variant>/ttgir
            found.append((flat_op, name, os.path.join(top, "ttgir")))
            continue
        for variant in sorted(os.listdir(top)):             # <root>/<op>/<variant>/ttgir
            sub = os.path.join(top, variant)
            if os.path.isdir(os.path.join(sub, "ttgir")):
                found.append((name, variant, os.path.join(sub, "ttgir")))
    return found


def collect(root, kernel_filter=None, dtype_filter=()):
    """op -> variant -> rel -> analysis, plus op -> sorted rels and the dropped count."""
    analyses = defaultdict(lambda: defaultdict(dict))
    paths = defaultdict(set)
    dropped = 0
    for op, variant, ttgir in matrices(root):
        for dirpath, _, names in os.walk(ttgir):
            for name in names:
                if not name.endswith(".ttgir"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ttgir).replace("\\", "/")
                if kernel_filter and kernel_filter not in rel:
                    continue
                if not dtype_matches(rel, dtype_filter):
                    dropped += 1
                    continue
                analyses[op][variant][rel] = analyse(full)
                paths[op].add(rel)
    return analyses, {op: sorted(rels) for op, rels in paths.items()}, dropped


def diff_size(a, b):
    """Normalised line-diff size between two analyses."""
    la, lb = a["lines"], b["lines"]
    if la == lb:
        return 0
    # cheap multiset difference: enough to say "changed or not" and how much
    ca, cb = Counter(la), Counter(lb)
    return sum((ca - cb).values()) + sum((cb - ca).values())


def fingerprint_groups(variants, rel):
    groups = defaultdict(list)
    for variant, files in variants.items():
        if rel in files:
            groups[files[rel]["fingerprint"]].append(variant)
    return groups


def column_order(names, chain):
    """Chain variants first, in dependency order, then anything else alphabetically."""
    head = [v for v in chain if v in names]
    return head + sorted(v for v in names if v not in head)


def report_op(op, variants, rels, chain, base, multi):
    """Print one op's tables and marginal attribution; return its CSV rows."""
    names = column_order(variants, chain)
    if multi:
        print("")
        print("### %s (%d variant(s))" % (op, len(names)))
    print("variants (%d), in column order: %s" % (len(names), ", ".join(names)))
    print("reference  : %s" % base)

    rows = []
    for rel in rels:
        print("")
        print("== %s ==" % (("%s/%s" % (op, rel)) if multi else rel))
        print("   %-16s %8s %12s  %s"
              % ("variant", "converts", "diff_vs_base", "layout fingerprint"))
        for name in names:
            got = variants[name].get(rel)
            if not got:
                print("   %-16s %8s %12s  %s" % (name, "-", "-", "(missing in this variant)"))
                continue
            delta = diff_size(variants[base][rel], got) if rel in variants[base] else 0
            print("   %-16s %8d %12d  %s" % (name, got["converts"], delta,
                                             got["fingerprint"]))
        groups = fingerprint_groups(variants, rel)
        if len(groups) > 1:
            print("   -- equivalent variants (identical layout set) --")
            for fingerprint, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                print("      [%d variant(s)] %s <- %s"
                      % (len(members), fingerprint, ", ".join(members)))
        else:
            print("   -- all variants share one layout set --")
        row = {"op": op, "path": rel}
        for name in names:
            got = variants[name].get(rel)
            row[name] = got["converts"] if got else ""
        rows.append(row)

    print("\n== marginal attribution along the dependency chain ==")
    chain_found = [v for v in chain if v in variants]
    if len(chain_found) < 2:
        print("  (need at least two chain variants; have %s)" % ", ".join(chain_found))
    for left, right in zip(chain_found, chain_found[1:], strict=False):
        what = SWITCH_OF_STEP.get((left, right), "%s -> %s" % (left, right))
        changed = []
        for rel in rels:
            a, b = variants[left].get(rel), variants[right].get(rel)
            if a and b and diff_size(a, b) > 0:
                changed.append("%s (converts %d->%d)" % (rel, a["converts"], b["converts"]))
        print("  %-38s %s" % (what, "; ".join(changed) if changed else "no change"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding <variant>/ttgir/... or "
                                 "<op>/<variant>/ttgir/...")
    ap.add_argument("--op", action="append", default=[], metavar="NAME[,NAME]",
                    help="only these ops of a multi-op root (default: all of them)")
    ap.add_argument("--kernel", default=None, help="only paths containing this substring")
    ap.add_argument("--dtype", action="append", default=[], metavar="DTYPE[,DTYPE]",
                    help="only files whose name carries this dtype "
                         "(fp16/bf16/fp32/..., aliases welcome; default: every dtype)")
    ap.add_argument("--base", default=None,
                    help="reference variant for the diff column (default: first in the chain)")
    ap.add_argument("--chain", default=None,
                    help="comma separated variant order for the marginal attribution "
                         "(default: %s)" % ", ".join(CHAIN))
    ap.add_argument("--csv", default=None, help="also write the matrix as CSV")
    args = ap.parse_args(argv)

    chain = [v.strip() for v in (args.chain or ",".join(CHAIN)).split(",") if v.strip()]
    tokens = dtype_tokens(args.dtype)
    analyses, paths, dropped = collect(args.root, args.kernel, tokens)
    if not analyses:
        print("!! no <variant>/ttgir/**.ttgir under %s" % args.root, file=sys.stderr)
        return 2

    wanted = split_names(args.op)
    ops = [op for op in sorted(analyses) if not wanted or op in wanted]
    if not ops:
        print("!! no matrix for op(s) %s under %s; have: %s"
              % (", ".join(sorted(wanted)), args.root, ", ".join(sorted(analyses))),
              file=sys.stderr)
        return 2
    if dropped:
        print("(dtype filter %s: %d ttgir file(s) left out)"
              % ("/".join(sorted(tokens)), dropped))

    multi = len(ops) > 1
    rows = []
    for op in ops:
        variants = analyses[op]
        base = args.base or next((v for v in chain if v in variants),
                                 column_order(variants, chain)[0])
        if base not in variants:
            print("!! base variant %r not found for %s; have: %s"
                  % (base, op, ", ".join(variants)), file=sys.stderr)
            return 2
        rows += report_op(op, variants, paths[op], chain, base, multi)

    if args.csv:
        names = column_order({name for files in analyses.values() for name in files}, chain)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["op", "path"] + names, restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print("\nmatrix written to %s" % args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
