"""compare_ttgir_matrix.py: name the switch that moved the layout, per op and dtype.

The IR here is the smallest module the analyser understands — one or two `#ttg.blocked`
definitions plus N `ttg.convert_layout` lines — written into a temp dir, so each layout
case is spelled out instead of hidden in a checked-in dump. The switch chain is the
real one: `enhance_off -> c0b0s0r0 -> c1b0s0r0 -> c1b1s0r0 -> c1b1s1r0 -> c1b1s1r1`.
"""
import contextlib
import csv
import io
import os

import compare_ttgir_matrix as cm

SHAPE = "64x64"
KERNEL = "_kernel"
FP16 = "w8s3__fp16.ttgir"
BF16 = "w8s3__bf16.ttgir"

LAYOUT = ("#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], "
          "warpsPerCTA = [8], order = [0]}>")
LAYOUT_WIDE = ("#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], "
               "warpsPerCTA = [8], order = [0]}>")

# variant -> (ttg.convert_layout count, also uses the [4] layout)
VARIANTS = {
    "enhance_off": (0, False),
    "c0b0s0r0": (0, False),
    "c1b0s0r0": (0, False),
    "c1b1s0r0": (3, True),
    "c1b1s1r0": (3, True),
    "c1b1s1r1": (3, True),
}


def _module(converts, wide):
    """One TTGIR module: `converts` convert_layouts, the [4] layout only when `wide`."""
    lines = [LAYOUT] + ([LAYOUT_WIDE] if wide else [])
    lines.append("module {")
    target = "#blocked1" if wide else "#blocked"
    for i in range(converts):
        lines.append("  %%v%d = ttg.convert_layout %%s%d : tensor<64x64xf16, #blocked> -> "
                     "tensor<64x64xf16, %s>" % (i, i, target))
    lines += ["  %c0 = arith.constant 0 : i32", "  %c1 = arith.constant 1 : i32", "}"]
    return "\n".join(lines) + "\n"


def _matrix(root, filename=FP16, op=None, variants=None):
    """Write <root>[/<op>]/<variant>/ttgir/<shape>/<kernel>/<file> for each variant."""
    base = os.path.join(root, op) if op else root
    for variant, (converts, wide) in (variants or VARIANTS).items():
        path = os.path.join(base, variant, "ttgir", SHAPE, KERNEL)
        os.makedirs(path)
        with open(os.path.join(path, filename), "w", encoding="utf-8") as fh:
            fh.write(_module(converts, wide))
    return root


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cm.main(argv)
    return rc, buf.getvalue()


def test_attributes_the_convert_increase_to_backward_propagation(tmp_path):
    """A convert appearing when backward_propagation turns on must be credited to that
    switch, and every other step must read `no change`."""
    rc, out = _run([_matrix(str(tmp_path))])
    assert rc == 0
    assert "backward_propagation 0->1" in out
    assert "converts 0->3" in out
    for quiet in ("master(=1) + cost_based off", "cost_based 0->1",
                  "small_component_solving 0->1", "store_layout_remat 0->1"):
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(quiet))
        assert line.rstrip().endswith("no change"), line


def test_column_order_puts_the_baseline_first(tmp_path):
    """enhance_off is the A side: it heads the table instead of sorting last."""
    _, out = _run([_matrix(str(tmp_path))])
    head = next(ln for ln in out.splitlines() if ln.startswith("variants ("))
    assert head.split(": ", 1)[1].startswith(
        "enhance_off, c0b0s0r0, c1b0s0r0, c1b1s0r0, c1b1s1r0, c1b1s1r1"), head
    assert "reference  : enhance_off" in out


def test_fingerprint_groups_the_equivalent_variants(tmp_path):
    """Three variants share the [1]-only layout set and three the wide one; the grouping
    is what says "equivalent" without reading every file."""
    _, out = _run([_matrix(str(tmp_path))])
    assert "[3 variant(s)]" in out
    assert "b[[1]|[32]|[8]|[0]] x" in out
    assert "b[[4]|[32]|[8]|[0]] x" in out


def test_multi_op_root_reports_each_op(tmp_path):
    """gen_combination_ttgir.sh writes <root>/<op>/<variant>/ttgir: the op directory is
    an op, not a variant."""
    _matrix(str(tmp_path), op="coalesce")
    _matrix(str(tmp_path), op="zeros", filename=BF16)
    rc, out = _run([str(tmp_path)])
    assert rc == 0
    assert "### coalesce (6 variant(s))" in out
    assert "### zeros (6 variant(s))" in out
    assert "== coalesce/64x64/_kernel/w8s3__fp16.ttgir ==" in out
    assert "== zeros/64x64/_kernel/w8s3__bf16.ttgir ==" in out


def test_op_filter_selects_one_op(tmp_path):
    _matrix(str(tmp_path), op="coalesce")
    _matrix(str(tmp_path), op="zeros", filename=BF16)
    rc, out = _run([str(tmp_path), "--op", "zeros"])
    assert rc == 0
    assert "### coalesce" not in out
    assert "== coalesce/" not in out
    assert "w8s3__bf16.ttgir" in out


def test_dtype_filter_reads_one_dtype(tmp_path):
    """The matrix is generated for one dtype by default; the other dtypes' dumps in the
    same tree must not be mixed in when one is asked for."""
    _matrix(str(tmp_path), op="coalesce")
    _matrix(str(tmp_path), op="zeros", filename=BF16)
    rc, out = _run([str(tmp_path), "--dtype", "bf16"])
    assert rc == 0
    assert "w8s3__bf16.ttgir" in out
    assert "w8s3__fp16.ttgir" not in out
    assert "dtype filter" in out          # says how many files the filter left out
    # a dtype an op was never dumped for must fail, not print an empty table
    rc, _ = _run([str(tmp_path), "--op", "coalesce", "--dtype", "bf16"])
    assert rc == 2


def test_kernel_filter_and_csv(tmp_path):
    path = os.path.join(str(tmp_path), "matrix.csv")
    root = _matrix(str(tmp_path))
    rc, _ = _run([root, "--kernel", KERNEL, "--csv", path])
    assert rc == 0
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["op"] == os.path.basename(os.path.normpath(root))
    assert rows[0]["c1b1s0r0"] == "3"
    assert rows[0]["enhance_off"] == "0"


def test_unknown_chain_is_not_an_error_but_missing_root_is(tmp_path):
    """A matrix with renamed variants (no chain names) still compares; only a root with
    no ttgir at all is an error."""
    root = _matrix(str(tmp_path), variants={"off": (0, False), "on": (2, True)})
    rc, out = _run([root])
    assert rc == 0
    assert "variants (2)" in out and "on -> " not in out
    rc, _ = _run([os.path.join(str(tmp_path), "does-not-exist")])
    assert rc == 2
