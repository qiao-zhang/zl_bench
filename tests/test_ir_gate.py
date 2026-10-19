import ir_gate

TTGIR = """\
#loc = loc("k.py":1:0)
module {
  %0 = arith.constant 1 : i32 loc(#loc1)
  %1 = arith.addi %0, %0 : i32 loc(#loc2)
}
"""


def test_normalize_ignores_loc_ssa_names_and_order():
    renamed = TTGIR.replace("%0", "%a").replace("%1", "%b").replace("loc(#loc2)", "loc(#loc9)")
    lines = TTGIR.splitlines()
    reordered = "\n".join(lines[:2] + [lines[3], lines[2]] + lines[4:])
    assert ir_gate._normalize(TTGIR, exact=False) == ir_gate._normalize(renamed, exact=False)
    assert ir_gate._normalize(TTGIR, exact=False) == ir_gate._normalize(reordered, exact=False)
    assert ir_gate._normalize(TTGIR, exact=True) != ir_gate._normalize(renamed, exact=True)
    changed = TTGIR.replace("arith.addi", "arith.muli")
    assert ir_gate._normalize(TTGIR, exact=False) != ir_gate._normalize(changed, exact=False)


def _run_dir(root, kernels):
    """A run dir with ttir/<hash>/<kernel>.{ttir,json} and a matching index.tsv."""
    rows = ["shape\tkernel\tfile\tlaunches\tsweep_launches\tnum_warps\tnum_stages\t"
            "constexprs\tcache_hash"]
    for h, kernel, launches in kernels:
        d = root / "ttir" / h
        d.mkdir(parents=True)
        (d / f"{kernel}.ttir").write_text("module {}")
        (d / f"{kernel}.json").write_text("{}")
        rows.append(f"4x4\t{kernel}\tw4s3.ttgir\t{launches}\t0\t4\t3\t{{}}\t{h}")
    (root / "ttgir").mkdir()
    (root / "ttgir" / "index.tsv").write_text("\n".join(rows) + "\n")
    return root


def test_collect_dedups_by_cache_hash_and_labels_batch_ops(tmp_path):
    batch = tmp_path / "batch"
    (batch / "main").mkdir(parents=True)
    _run_dir(batch / "main" / "abs", [("H1", "abs_kernel", 2), ("H2", "fill_kernel", 1)])
    _run_dir(batch / "main" / "neg", [("H2", "fill_kernel", 1)])
    (batch / "ops_status.csv").write_text(
        "repo,op,op_file,run_name\nmain,abs,abs,abs\nmain,neg,neg,neg\n")
    items = ir_gate.collect([batch])
    assert sorted(v.ttir.parent.name for v in items) == ["H1", "H2"]
    assert {v.label for v in items} == {"main:abs@abs"}
    assert all(v.recorded and v.recorded.name == "w4s3.ttgir" for v in items)


def test_collect_single_run_dir(tmp_path):
    run = _run_dir(tmp_path / "softmax_run", [("H1", "softmax_kernel", 1)])
    (item,) = ir_gate.collect([run])
    assert item.label == "softmax_run"
    assert str(item.key) == "H1/softmax_kernel.ttgir"


def _digest(ttgir="t1", norm="n1", cubin="c1", shared=0):
    return {"ttgir": ttgir, "ttgir_norm": norm, "ops": {}, "llir": "l1", "ptx": "p1",
            "cubin": cubin, "meta": {"shared": shared, "num_warps": 4}}


def test_compare_finds_the_first_differing_stage():
    base = _digest()
    assert ir_gate.compare(base, _digest(), "cubin") == ("0", "-", "same")
    # same TTGIR, different binary: only the cubin gate sees it
    assert ir_gate.compare(base, _digest(cubin="c2"), "cubin") == ("1", "cubin", "same")
    assert ir_gate.compare(base, _digest(cubin="c2"), "ttgir") == ("0", "-", "same")
    assert ir_gate.compare(base, _digest(ttgir="t2"), "cubin") == ("1", "ttgir", "order")
    assert ir_gate.compare(base, _digest(ttgir="t2", norm="n2"), "cubin") == ("1", "ttgir", "changed")
    assert ir_gate.compare(base, _digest(shared=1024), "cubin") == ("1", "meta", "same")
    assert ir_gate.compare(base, None, "cubin") == ("?", "error", "?")


def test_run_dirs_and_rows(tmp_path):
    batch = tmp_path / "batch"
    run = _run_dir(batch / "main" / "abs", [("H1", "abs_kernel", 2)])
    (batch / "ops_status.csv").write_text("repo,op,op_file,run_name\nmain,abs,abs,abs\n")
    (run / "rows.tsv").write_text("op_name\tdtype\trow\tshape\tkernels\nabs\ttorch.float16\t0\t4x4\tH1\n")
    (rd,) = ir_gate.run_dirs([batch])
    assert (rd.label, rd.run) == ("main:abs@abs", "main/abs")
    assert ir_gate.run_hashes(rd.path) == {"H1"}
    assert ir_gate.read_rows(rd.path)[0]["kernels"] == "H1"
    assert ir_gate.read_rows(tmp_path) is None


def test_op_counts_and_diff():
    base = ir_gate.op_counts(TTGIR)
    assert base == {"arith.constant": 1, "arith.addi": 1}
    new = ir_gate.op_counts(TTGIR.replace("%1 = arith.addi %0, %0", '%1:2 = "tt.foo.bar"(%0)')
                            + "  ttg.local_load %0 : i32\n  ttg.local_load %1 : i32\n")
    assert new["tt.foo.bar"] == 1 and new["ttg.local_load"] == 2
    diff = ir_gate.op_diff({"ops": base}, {"ops": new})
    assert diff == "ttg.local_load:+2 arith.addi:-1 tt.foo.bar:+1"
    assert ir_gate.op_diff(None, {"ops": new}) == ""


def _hgbin_digest(hgbin="b1", llir="l1", ttgir="t1", norm="n1", shared=0):
    return {"ttgir": ttgir, "ttgir_norm": norm, "ops": {}, "llir": llir, "hgbin": hgbin,
            "meta": {"shared": shared, "num_warps": 4},
            "stages": ["ttgir", "llir", "hgbin"], "binary": "hgbin"}


def test_compare_follows_the_backends_own_stage_chain():
    """PPU records ttgir/llir/hgbin: the gate must compare hgbin, never look for ptx/cubin."""
    base = _hgbin_digest()
    assert ir_gate.binary_stage(base) == "hgbin"
    assert ir_gate.stage_chain(base) == ["ttgir", "llir", "hgbin"]
    assert ir_gate.compare(base, _hgbin_digest(), "binary") == ("0", "-", "same")
    assert ir_gate.compare(base, _hgbin_digest(hgbin="b2"), "binary") == ("1", "hgbin", "same")
    assert ir_gate.compare(base, _hgbin_digest(llir="l2"), "binary") == ("1", "llir", "same")
    assert ir_gate.compare(base, _hgbin_digest(ttgir="t2"), "binary") == ("1", "ttgir", "order")
    assert ir_gate.compare(base, _hgbin_digest(shared=1024), "binary") == ("1", "meta", "same")
    # --stage ttgir stops before the hgbin, so a binary-only change is invisible
    assert ir_gate.compare(base, _hgbin_digest(hgbin="b2"), "ttgir") == ("0", "-", "same")


def test_old_digests_without_a_chain_keep_nvidias_stages():
    base = _digest()
    assert ir_gate.stage_chain(base) == list(ir_gate.STAGES)
    assert ir_gate.binary_stage(base) == "cubin"
