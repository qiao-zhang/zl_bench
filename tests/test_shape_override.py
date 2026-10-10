import _shape_override_plugin as plugin


class FakeBenchmark:
    def __init__(self, op_name, shapes):
        self.op_name, self._shapes = op_name, shapes

    def init_user_config(self):
        self.shapes = list(self._shapes)  # what a hardcoded set_shapes() leaves behind


def _patched(monkeypatch, entries):
    monkeypatch.setattr(plugin, "_entries", entries)

    class Bench(FakeBenchmark):
        pass

    plugin._patch_init_user_config(Bench)
    return Bench


def test_yaml_shapes_override_hardcoded(monkeypatch):
    Bench = _patched(monkeypatch, {"moe": {"shapes": [[1, 2, 3]], "shape_desc": "M, N, K"}})
    b = Bench("moe", [(9, 9, 9), (8, 8, 8)])
    b.init_user_config()
    assert b.shapes == [(1, 2, 3)]
    assert b.shape_desc == "M, N, K"


def test_ops_without_entry_keep_their_shapes(monkeypatch):
    Bench = _patched(monkeypatch, {"other": {"shapes": [[1]]}})
    b = Bench("moe", [(9, 9, 9)])
    b.init_user_config()
    assert b.shapes == [(9, 9, 9)]


def test_arity_normalization_is_preserved(monkeypatch):
    Bench = _patched(monkeypatch, {"mm": {"shapes": [[2, 64, 64, 64]]}})
    b = Bench("mm", [(64, 64, 64)])  # subclass normalized (B, M, N, K) -> (M, N, K)
    b.init_user_config()
    assert b.shapes == [(64, 64, 64)]


def test_standard_input_fn_signature():
    assert plugin._takes_shape_dtype_device(lambda shape, dtype, device: None)
    assert not plugin._takes_shape_dtype_device(lambda b, m, n, k, dtype, device, t: None)
    assert not plugin._takes_shape_dtype_device(None)
