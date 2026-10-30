"""The layout audit is a diagnostic, but its failure modes are silent: a basis it
cannot parse underestimates the footprint, a renamed alias or a total-count
comparison hides a hit that moved, and a max() comparison hides a larger hit that
vanished while a smaller one grew. All of them are pinned here."""

import ir_layout_audit as audit


LINEAR_8X = ("#ttg.linear<{register = [[0, 1], [0, 2], [0, 32]], "
             "lane = [[0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 16]], "
             "block = []}>")
LINEAR_2X = ("#ttg.linear<{register = [[0, 8]], lane = [[0, 0], [0, 0], [0, 1], "
             "[0, 2], [0, 4]], warp = [[0, 16]], block = []}>")
LINEAR_1X = ("#ttg.linear<{register = [], lane = [[0, 1], [0, 2], [0, 4], [0, 8], "
             "[0, 16]], warp = [[0, 32]], block = []}>")


def test_basis_vectors_accepts_any_rank_and_never_filters_zero_vectors():
    assert audit.basis_vectors("[[1], [2]]") == 2                 # 1-D vectors
    assert audit.basis_vectors("[[0, 0], [0, 1]]") == 2           # zero vector still counts
    assert audit.basis_vectors("[[0, 0, 1], [0, 0, 0]]") == 2     # 3-D vectors
    assert audit.basis_vectors("[]") == 0


def test_basis_components_are_read_per_name():
    assert audit.basis_vectors(audit.component(LINEAR_8X, "register")) == 3
    assert audit.basis_vectors(audit.component(LINEAR_8X, "lane")) == 5
    assert audit.zero_bases(audit.component(LINEAR_8X, "lane")) == 3
    assert audit.basis_vectors(audit.component(LINEAR_1X, "register")) == 0


def _write(directory, layouts, shape="1x64", warps=2, threads=32, name="k.ttgir"):
    """Write a TTGIR with one load per (alias, body); a plain string means one `#linear`."""
    if isinstance(layouts, str):
        layouts = [("#linear", layouts)]
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{alias} = {body}" for alias, body in layouts]
    lines.append(f'module attributes {{"ttg.num-warps" = {warps} : i32, '
                 f'"ttg.threads-per-warp" = {threads} : i32}} {{')
    for i, (alias, _) in enumerate(layouts):
        lines.append(f"  %p{i} = tt.splat %x : !tt.ptr<f32> -> "
                     f"tensor<{shape}x!tt.ptr<f32>, {alias}>")
        lines.append(f"  %v{i} = tt.load %p{i} : tensor<{shape}x!tt.ptr<f32>, {alias}>")
    lines.append("}")
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory / name


def test_scan_flags_the_eight_way_footprint(tmp_path):
    hits, threads = audit.scan(_write(tmp_path / "main" / "kron", LINEAR_8X), "main/kron/k.ttgir")
    assert threads == 64
    assert [(h.shape, h.elems_per_thread, h.duplication) for h in hits] == [("1x64", 8, 8.0)]
    assert hits[0].zero_lane_bases == 3


def test_scan_leaves_a_matching_footprint_alone(tmp_path):
    hits, _ = audit.scan(_write(tmp_path / "main" / "kron", LINEAR_1X), "main/kron/k.ttgir")
    assert hits == []


def test_a_hit_that_moves_between_kernels_is_not_a_no_op(tmp_path):
    root = tmp_path / "gate"
    _write(root / "base" / "main" / "kron", LINEAR_8X, name="a.ttgir")
    _write(root / "new" / "main" / "kron", LINEAR_8X, name="b.ttgir")
    sides = audit.collect(root)
    added, removed, worsened = audit.diff_sides(sides["base"], sides["new"])
    # Same total (1 vs 1), different kernel: both halves have to be reported.
    assert len(sides["base"]) == len(sides["new"]) == 1
    assert sum(added.values()) == 1 and sum(removed.values()) == 1
    assert worsened == []


def test_a_hit_that_grows_in_place_is_reported_as_worsened(tmp_path):
    root = tmp_path / "gate"
    _write(root / "base" / "main" / "kron", LINEAR_2X, name="k.ttgir")
    _write(root / "new" / "main" / "kron", LINEAR_8X, name="k.ttgir")
    added, removed, worsened = audit.diff_sides(*audit.collect(root).values())
    # the path separator is platform-dependent, so compare the numbers and the tail
    assert [(shape, before, after) for (_, shape), before, after in worsened] == \
        [("1x64", 2.0, 8.0)]
    assert worsened[0][0][0].endswith("k.ttgir")
    assert sum(added.values()) == 1 and sum(removed.values()) == 1


def test_a_renamed_alias_is_not_a_change(tmp_path):
    """MLIR numbers aliases by order of appearance, so one layout can be `#a` on one
    side and `#b` on the other. Matching on the alias would report a phantom pair."""
    root = tmp_path / "gate"
    _write(root / "base" / "main" / "kron", [("#a", LINEAR_8X)], name="k.ttgir")
    _write(root / "new" / "main" / "kron", [("#b", LINEAR_8X)], name="k.ttgir")
    sides = audit.collect(root)
    added, removed, worsened = audit.diff_sides(sides["base"], sides["new"])
    assert not added and not removed and not worsened
    assert audit.distribution_changes(sides["base"], sides["new"]) == []


def test_vanishing_large_hit_does_not_read_as_no_change(tmp_path):
    """base: 8x and 2x hits in one kernel. new: the 8x hit is gone and the remaining one
    reads 4x (more warps). A max() comparison sees 8x -> 4x and calls it an improvement,
    while the kept hit actually got worse — so this must not report "no new duplication"."""
    root = tmp_path / "gate"
    _write(root / "base" / "main" / "kron",
           [("#a", LINEAR_8X), ("#b", LINEAR_2X)], threads=32, name="k.ttgir")
    _write(root / "new" / "main" / "kron",
           [("#b", LINEAR_2X)], warps=4, threads=32, name="k.ttgir")
    sides = audit.collect(root)
    added, removed, worsened = audit.diff_sides(sides["base"], sides["new"])
    changed = audit.distribution_changes(sides["base"], sides["new"])
    assert sum(added.values()) == 1      # the 2x -> 4x hit, visible because the key has dup
    assert len(changed) == 1             # the 8x hit vanished: the distribution differs
    assert changed[0][2] == {8.0: 1, 2.0: 1}
    assert changed[0][3] == {4.0: 1}
    assert worsened == []                # max() alone would have read this as an improvement


def test_a_pure_removal_is_not_reported_as_no_change(tmp_path):
    """A hit disappearing is still a difference: `no change` would contradict the
    promise that only an identical pair of sides reads that way."""
    root = tmp_path / "gate"
    _write(root / "base" / "main" / "kron", LINEAR_8X, name="k.ttgir")
    _write(root / "new" / "main" / "kron", LINEAR_1X, name="k.ttgir")
    sides = audit.collect(root)
    added, removed, worsened = audit.diff_sides(sides["base"], sides["new"])
    changed = audit.distribution_changes(sides["base"], sides["new"])
    assert sum(removed.values()) == 1 and not added and not worsened
    assert changed == []                      # the location is gone, not shifted
    assert audit.classify(added, removed, worsened, changed) == "reduced"


def test_summary_classification_covers_all_four_cases():
    one = audit.Counter({("f", "l", "1x64", 8, 8.0): 1})
    none = audit.Counter()
    assert audit.classify(one, none, [], []) == "regression"
    assert audit.classify(none, none, [(("f", "1x64"), 2.0, 4.0)], []) == "regression"
    assert audit.classify(none, one, [], [("f", "1x64", {}, {})]) == "review"
    assert audit.classify(none, one, [], []) == "reduced"
    assert audit.classify(none, none, [], []) == "no change"


def test_threads_come_from_the_module_attributes(tmp_path):
    hits, threads = audit.scan(
        _write(tmp_path / "main" / "kron", LINEAR_8X, warps=4, threads=32), "main/kron/k.ttgir")
    assert threads == 128
    # 8 elems/thread * 128 threads / 64 elements = 16x, not the hard-coded 8x
    assert hits and hits[0].duplication == 16.0
