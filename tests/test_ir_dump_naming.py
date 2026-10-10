import _ir_dump_plugin as dump


def test_abbrev_and_fmt_value():
    assert dump._abbrev("BLOCK_SIZE_M") == "BSM"
    assert dump._abbrev("BLOCK_M2") == "BM2"
    assert dump._abbrev("SPLIT") == "SPLIT"
    assert dump._fmt_value(True) == "T"
    assert dump._fmt_value(1) == "1"  # not "T": 1 == True in Python
    assert dump._fmt_value(None) == "-"
    assert dump._fmt_value("triton.language.bfloat16") == "bfloat16"


def _variant(h, warps=4, stages=3, signature=None, **constexprs):
    return {"hash": h, "warps": warps, "stages": stages, "constexprs": constexprs,
            "signature": signature or {}, "attrs": {}}


def test_name_group_uses_only_varying_constexprs():
    items = [_variant("AAAA1111", BLOCK_SIZE_M=64, GROUP=8),
             _variant("BBBB2222", BLOCK_SIZE_M=128, GROUP=8)]
    named, legend = dump._name_group(items)
    assert sorted(name for _, name in named) == ["BSM128_w4s3.ttgir", "BSM64_w4s3.ttgir"]
    assert legend == {"BLOCK_SIZE_M": "BSM"}


def test_name_group_disambiguates_by_dtype_then_hash():
    by_dtype = [_variant("AAAA1111", signature={"x": "*fp16"}, B=1),
                _variant("BBBB2222", signature={"x": "*bf16"}, B=1)]
    named, _ = dump._name_group(by_dtype)
    assert sorted(name for _, name in named) == ["w4s3__bf16.ttgir", "w4s3__fp16.ttgir"]

    by_hash = [_variant("AAAA1111", B=1), _variant("BBBB2222", B=1)]
    named, _ = dump._name_group(by_hash)
    assert sorted(name for _, name in named) == ["w4s3__AAAA1111.ttgir", "w4s3__BBBB2222.ttgir"]


def test_rows_tsv_prefers_the_timed_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(dump, "_tables",
                        [("softmax", "torch.float16", [(0, "4x4"), (1, "8x8"), (None, "x")])])
    monkeypatch.setattr(dump, "_row_timed", {0: {"H2", "H1"}})
    monkeypatch.setattr(dump, "_row_any", {0: {"H1", "H2", "H3"}, 1: {"H4"}})
    dump._write_rows(tmp_path / "rows.tsv")
    lines = (tmp_path / "rows.tsv").read_text().splitlines()
    assert lines[0] == "op_name\tdtype\trow\tshape\tkernels\tscope"
    assert lines[1:] == ["softmax\ttorch.float16\t0\t4x4\tH1,H2\ttimed",
                         "softmax\ttorch.float16\t1\t8x8\tH4\trow",
                         "softmax\ttorch.float16\t2\tx\t?\t-"]
