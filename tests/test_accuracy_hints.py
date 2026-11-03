import argparse
import json

import pytest

import ab_run
import accuracy_hints
import run_batch


def _checkout(tmp_path, label="main"):
    root = tmp_path / label
    (root / "tests").mkdir(parents=True)
    return root


def _write(root, rel, text):
    path = root / "tests" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _index(root):
    return accuracy_hints.TestsIndex(root)


def test_suggest_lists_the_same_file_when_the_function_name_differs(tmp_path):
    """The gate looks for `test_<op>`; upstream often splits the op into several cases
    (`test_kda_gate`, `test_kda_state`). The same file is the best candidate there is."""
    root = _checkout(tmp_path)
    _write(root, "test_kda.py", "def test_kda_gate():\n    pass\n"
                                "def test_kda_state():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "kda", "kda")
    assert [(h.tier, h.tag) for h in hints] == [(1, "同名文件"), (1, "同名文件")]
    assert {h.target for h in hints} == {"tests/test_kda.py::test_kda_gate",
                                         "tests/test_kda.py::test_kda_state"}
    assert "只是不叫 `test_kda`" in hints[0].why


def test_suggest_falls_back_to_a_name_similar_file(tmp_path):
    """`tests/test_FLA/chunk_gdn2.py` does not exist, but `test_chunk_gdn2.py` does, and
    its case carries the op name."""
    root = _checkout(tmp_path)
    _write(root, "test_FLA/test_chunk_gdn2.py",
           "def test_chunk_gdn2_matches_native_triton():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "chunk_gdn2", "FLA/chunk_gdn2")
    assert [h.target for h in hints] == [
        "tests/test_FLA/test_chunk_gdn2.py::test_chunk_gdn2_matches_native_triton"]
    assert hints[0].tier in (2, 3)


def test_suggest_matches_the_function_when_the_file_name_says_nothing(tmp_path):
    """`abs` has no file of its own; the case is named after the op inside another one."""
    root = _checkout(tmp_path)
    _write(root, "test_misc.py", "def test_abs_accuracy():\n    pass\n"
                                 "def test_other():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "abs", "abs_inplace")
    assert [h.target for h in hints] == ["tests/test_misc.py::test_abs_accuracy"]
    assert hints[0].tier == 4


def test_suggest_uses_the_benchmark_name_as_a_last_resort(tmp_path):
    """`sparse_attn_triton` lives in `benchmark/test_sparse_attention.py`; the test is
    named after that file, so the op name alone finds nothing."""
    root = _checkout(tmp_path)
    _write(root, "test_flash_attention.py", "def test_sparse_attention():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "sparse_attn_triton", "sparse_attention")
    assert [h.target for h in hints] == ["tests/test_flash_attention.py::test_sparse_attention"]
    assert hints[0].tier == 5
    assert "确认它是否覆盖这个算子" in hints[0].why


def test_suggest_gives_a_class_method_its_class_path(tmp_path):
    """Upstream often collects a whole file of class-based cases (`TestGridSample...`):
    a bare `::test_accuracy` is a node pytest cannot run, and the map would fail the op."""
    root = _checkout(tmp_path)
    _write(root, "test_grid_sample.py",
           "class TestGridSampleNearest4D:\n"
           "    def test_nearest_small(self):\n        pass\n"
           "class TestGridSampleEdgeCases:\n"
           "    def test_edge(self):\n        pass\n")
    hints = accuracy_hints.suggest(_index(root), "grid_sample", "grid_sample")
    assert {h.target for h in hints} == {
        "tests/test_grid_sample.py::TestGridSampleNearest4D::test_nearest_small",
        "tests/test_grid_sample.py::TestGridSampleEdgeCases::test_edge"}
    assert all(h.tier == 1 for h in hints)


def test_test_nodes_keep_the_class_path_and_skip_nested_functions(tmp_path):
    """Only what pytest collects: module-level functions, class methods (every nesting
    level), and nothing defined inside another function."""
    root = _checkout(tmp_path)
    path = _write(root, "test_mixed.py",
                  "def test_top():\n    pass\n"
                  "def helper():\n"
                  "    def test_inner():\n        pass\n"
                  "class TestA:\n"
                  "    def test_method(self):\n        pass\n"
                  "    def test_not_a_case(self):\n        pass\n"
                  "    class TestNested:\n"
                  "        def test_deep(self):\n            pass\n")
    assert accuracy_hints.test_nodes(path) == [
        "test_top", "TestA::test_method", "TestA::test_not_a_case", "TestA::TestNested::test_deep"]


def test_suggest_keeps_an_exact_name_file_behind_six_near_misses(tmp_path):
    """The cap must not run in directory order: six name-similar files in the op's own
    directory used to hide the file whose name *is* the op."""
    root = _checkout(tmp_path)
    for i in range(6):
        _write(root, f"test_FLA/test_kda_a{i}.py", f"def test_kda_a{i}_case():\n    pass\n")
    _write(root, "test_kda.py", "def test_kda_case():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "kda", "FLA/kda")
    assert "tests/test_kda.py::test_kda_case" in {h.target for h in hints}
    assert hints[0].target == "tests/test_kda.py::test_kda_case"    # exact name first


def test_suggest_keeps_the_similarity_order_it_reports(tmp_path):
    """A same-file list is ordered by how close each name is to the op; re-sorting the
    result by target would quietly turn that into alphabetical order."""
    root = _checkout(tmp_path)
    _write(root, "test_abc_x.py", "def test_aaa():\n    pass\n"
                                  "def test_zabc():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "abc_x", "abc_x")
    assert [h.target for h in hints] == ["tests/test_abc_x.py::test_zabc",
                                         "tests/test_abc_x.py::test_aaa"]


def test_test_nodes_keep_a_def_under_a_condition_but_mark_it_uncertain(tmp_path):
    """pytest collects a `def` under `if True:` because it ran; under a branch that did
    not run it collects nothing. A static read cannot decide, so the node is kept with
    `certain=False` and only the certain ones go to the automatic lookup."""
    root = _checkout(tmp_path)
    path = _write(root, "test_cond.py",
                  "if True:\n    def test_softmax():\n        pass\n"
                  "try:\n    def test_guarded():\n        pass\n"
                  "except ImportError:\n    pass\n"
                  "def test_plain():\n    pass\n"
                  "with open('x') as f:\n    def test_in_with():\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_plain", "test_in_with"]
    info = {n.node: n.certain for n in accuracy_hints.test_node_info(path)}
    assert info == {"test_softmax": False, "test_guarded": False,
                    "test_plain": True, "test_in_with": True}


def test_suggest_marks_a_conditional_candidate(tmp_path):
    root = _checkout(tmp_path)
    _write(root, "test_softmax.py",
           "if some_flag:\n    def test_softmax_accuracy():\n        pass\n")
    hints = accuracy_hints.suggest(_index(root), "softmax", "softmax")
    assert [h.target for h in hints] == ["tests/test_softmax.py::test_softmax_accuracy"]
    assert hints[0].certain is False
    assert accuracy_hints._tag(hints[0]) == "同名文件·不确定"


def test_suggest_ignores_a_plain_helper_class(tmp_path):
    """pytest collects classes named `Test*` (and TestCase subclasses), not any class:
    `Helper::test_softmax` is a node it reports as not found (rc 4)."""
    root = _checkout(tmp_path)
    _write(root, "test_softmax.py",
           "class Helper:\n"
           "    def test_softmax(self):\n        pass\n"
           "\n"
           "def test_softmax_accuracy():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "softmax", "softmax")
    targets = {h.target for h in hints}
    assert "tests/test_softmax.py::Helper::test_softmax" not in targets
    assert "tests/test_softmax.py::test_softmax_accuracy" in targets


def test_test_nodes_follow_pytest_class_rules(tmp_path):
    """`Test*` and `unittest.TestCase` subclasses are collected; a class with an
    `__init__`, one that sets `__test__ = False`, and a plain helper are not."""
    root = _checkout(tmp_path)
    path = _write(root, "test_classes.py",
                  "import unittest\n"
                  "class TestPrefixed:\n"
                  "    def test_a(self):\n        pass\n"
                  "class NotPrefixed(unittest.TestCase):\n"
                  "    def test_b(self):\n        pass\n"
                  "class WithInit:\n"
                  "    def __init__(self):\n        pass\n"
                  "    def test_c(self):\n        pass\n"
                  "class OptedOut:\n"
                  "    __test__ = False\n"
                  "    def test_d(self):\n        pass\n"
                  "class Helper:\n"
                  "    def test_e(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["TestPrefixed::test_a",
                                               "NotPrefixed::test_b"]


def test_test_nodes_stop_on_a_module_level_opt_out(tmp_path):
    root = _checkout(tmp_path)
    path = _write(root, "test_off.py", "__test__ = False\n"
                                      "def test_a():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == []


def test_test_nodes_collect_a_testcase_that_defines_its_own_init(tmp_path):
    """`__init__` keeps pytest away from an ordinary `Test*` class, but a unittest class
    is driven through unittest's protocol, where its own `__init__` is the norm."""
    root = _checkout(tmp_path)
    path = _write(root, "test_unittest.py",
                  "import unittest\n"
                  "class Custom(unittest.TestCase):\n"
                  "    def __init__(self, methodName='runTest'):\n"
                  "        super().__init__(methodName)\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["Custom::test_softmax"]


def test_test_nodes_do_not_trust_a_testcase_name_suffix(tmp_path):
    """`class FakeTestCase` is a plain base: a name ending in `TestCase` is not evidence.
    The local class resolves it, so the subclass is no test either."""
    root = _checkout(tmp_path)
    path = _write(root, "test_fake.py",
                  "class FakeTestCase:\n    pass\n"
                  "class Helper(FakeTestCase):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_test_nodes_follow_a_unittest_import_alias(tmp_path):
    root = _checkout(tmp_path)
    path = _write(root, "test_alias.py",
                  "from unittest import TestCase as TC\n"
                  "class Custom(TC):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["Custom::test_softmax"]


def test_test_nodes_take_imported_subclasses_as_uncertain(tmp_path):
    """A TestCase subclass imported from another module cannot be resolved statically, so
    the class it is used in stays a maybe (`certain=False`)."""
    root = _checkout(tmp_path)
    path = _write(root, "test_base.py",
                  "from .base import BaseTestCase\n"
                  "class Custom(BaseTestCase):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Custom::test_softmax", False)]


def test_test_nodes_ignore_a_local_import_inside_a_function(tmp_path):
    """`from unittest import TestCase as Base` inside a function binds a local name; the
    module-level `class Base` it shadows is a plain class, so `Custom` is no test."""
    root = _checkout(tmp_path)
    path = _write(root, "test_shadow.py",
                  "def helper():\n"
                  "    from unittest import TestCase as Base\n"
                  "\n"
                  "class Base:\n    pass\n"
                  "\n"
                  "class Custom(Base):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_test_nodes_keep_a_conditional_import_uncertain(tmp_path):
    """A TestCase bound under `try`/`if` may or may not be that; the class stays a maybe
    (`certain=False`) instead of a certain source for the automatic lookup."""
    root = _checkout(tmp_path)
    path = _write(root, "test_cond_import.py",
                  "try:\n"
                  "    from unittest import TestCase as TC\n"
                  "except ImportError:\n"
                  "    TC = object\n"
                  "\n"
                  "class Custom(TC):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Custom::test_softmax", False)]


def test_test_nodes_drop_a_rebound_testcase_alias(tmp_path):
    """`class Base` after `from unittest import TestCase as Base` rebinds the name: a
    subclass of it is not a TestCase any more (pytest collects nothing)."""
    root = _checkout(tmp_path)
    path = _write(root, "test_rebound.py",
                  "from unittest import TestCase as Base\n"
                  "\n"
                  "class Base:\n    pass\n"
                  "\n"
                  "class Custom(Base):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert accuracy_hints.test_node_info(path) == []


def test_test_nodes_do_not_inherit_collectability_from_a_plain_test_base(tmp_path):
    """`TestBase` is collectable because of its name, and that says nothing about its
    subclass: `Helper(TestBase)` is neither a `Test*` class nor a TestCase, so pytest
    does not collect `Helper::test_softmax` (rc 4). Only the TestCase identity inherits."""
    root = _checkout(tmp_path)
    path = _write(root, "test_inherit.py",
                  "class TestBase:\n    pass\n"
                  "\n"
                  "class Helper(TestBase):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_test_nodes_inherit_the_testcase_identity(tmp_path):
    """A subclass of a real TestCase is collected whatever its name."""
    root = _checkout(tmp_path)
    path = _write(root, "test_case_base.py",
                  "import unittest\n"
                  "class Base(unittest.TestCase):\n    pass\n"
                  "class Helper(Base):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Helper::test_softmax", True)]


def test_a_plain_test_base_subclass_follows_the_ordinary_rules(tmp_path):
    """A `Test*` subclass of a plain class is collected; with its own `__init__` it is
    not (pytest warns and skips it)."""
    root = _checkout(tmp_path)
    ok = _write(root, "test_ok.py",
                "class TestBase:\n    pass\n"
                "class TestHelper(TestBase):\n"
                "    def test_a(self):\n        pass\n")
    assert accuracy_hints.test_nodes(ok) == ["TestHelper::test_a"]
    skipped = _write(root, "test_skipped.py",
                     "class TestBase:\n    pass\n"
                     "class TestHelper(TestBase):\n"
                     "    def __init__(self):\n        pass\n"
                     "    def test_a(self):\n        pass\n")
    assert accuracy_hints.test_nodes(skipped) == []


def test_test_nodes_skip_a_class_that_inherits_an_init(tmp_path):
    """pytest's `hasinit` looks at the whole class: `TestSoftmax(Base)` inherits Base's
    `__init__` and is skipped, so `TestSoftmax::test_softmax` is not a node (rc 4)."""
    root = _checkout(tmp_path)
    path = _write(root, "test_ctor.py",
                  "class Base:\n"
                  "    def __init__(self):\n        pass\n"
                  "\n"
                  "class TestSoftmax(Base):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_test_nodes_skip_a_class_that_inherits_a_new(tmp_path):
    """The same rule for `__new__`: pytest warns about a `__new__` constructor too."""
    root = _checkout(tmp_path)
    path = _write(root, "test_new.py",
                  "class WithNew:\n"
                  "    def __new__(cls):\n        return super().__new__(cls)\n"
                  "\n"
                  "class TestFromNew(WithNew):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]


def test_a_testcase_with_an_inherited_init_is_still_collected(tmp_path):
    """A unittest class ignores the constructor rule: both its own and an inherited
    `__init__` are normal there."""
    root = _checkout(tmp_path)
    path = _write(root, "test_tc_ctor.py",
                  "import unittest\n"
                  "class Base(unittest.TestCase):\n"
                  "    def __init__(self, methodName='runTest'):\n"
                  "        super().__init__(methodName)\n"
                  "class Helper(Base):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Helper::test_softmax", True)]


def test_test_nodes_skip_a_class_that_inherits_an_opt_out(tmp_path):
    """pytest reads `__test__` with `getattr`, so a subclass inherits `__test__ = False`
    from a plain base and is skipped even with a `Test*` name."""
    root = _checkout(tmp_path)
    path = _write(root, "test_optout.py",
                  "class Base:\n    __test__ = False\n"
                  "\n"
                  "class TestSoftmax(Base):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_a_subclass_can_re_enable_itself(tmp_path):
    """`__test__ = True` on the subclass overrides the inherited opt-out."""
    root = _checkout(tmp_path)
    path = _write(root, "test_reenable.py",
                  "class Base:\n    __test__ = False\n"
                  "\n"
                  "class TestReenabled(Base):\n"
                  "    __test__ = True\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["TestReenabled::test_softmax"]


def test_a_testcase_that_inherits_an_opt_out_is_skipped(tmp_path):
    """The flag wins over the unittest rules: a TestCase subclass below an opted-out base
    is not collected either."""
    root = _checkout(tmp_path)
    path = _write(root, "test_tc_optout.py",
                  "import unittest\n"
                  "class Base(unittest.TestCase):\n"
                  "    __test__ = False\n"
                  "class Helper(Base):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert accuracy_hints.test_node_info(path) == []


def test_an_unreadable_test_flag_leaves_the_class_uncertain(tmp_path):
    """`__test__ = FLAG` is not a literal: whether pytest skips the class depends on the
    value, so the node is a maybe rather than a certain target."""
    root = _checkout(tmp_path)
    path = _write(root, "test_dynamic.py",
                  "import os\n"
                  "FLAG = os.environ.get('X') == '1'\n"
                  "class TestSoftmax:\n"
                  "    __test__ = FLAG\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("TestSoftmax::test_softmax", False)]


def test_test_nodes_collect_a_plain_class_that_sets_test_true(tmp_path):
    """pytest takes a class when its name has the `Test*` prefix **or** it sets
    `__test__ = True`, so a helper class can opt in under any name."""
    root = _checkout(tmp_path)
    path = _write(root, "test_force.py",
                  "class Helper:\n"
                  "    __test__ = True\n"
                  "\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert {n.node for n in accuracy_hints.test_node_info(path)} == {
        "Helper::test_softmax", "test_softmax_accuracy"}


def test_the_last_test_flag_assignment_wins(tmp_path):
    """`__test__` is a plain attribute: the value the body ends with is pytest's, not the
    first one it sees."""
    root = _checkout(tmp_path)
    off = _write(root, "test_last_off.py",
                 "class TestSoftmax:\n"
                 "    __test__ = True\n"
                 "    __test__ = False\n"
                 "\n"
                 "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(off) == []
    on = _write(root, "test_last_on.py",
                "class TestSoftmax:\n"
                "    __test__ = False\n"
                "    __test__ = True\n"
                "\n"
                "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(on) == ["TestSoftmax::test_softmax"]


def test_a_conditional_test_flag_leaves_the_class_uncertain(tmp_path):
    """`if flag: __test__ = False` may or may not have run, so whether pytest skips the
    class depends on the value: the node is a maybe, not a target."""
    root = _checkout(tmp_path)
    path = _write(root, "test_cond_flag.py",
                  "FLAG = True\n"
                  "class TestCond:\n"
                  "    if FLAG:\n        __test__ = False\n"
                  "\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("TestCond::test_softmax", False), ("test_softmax_accuracy", True)]


def test_a_later_unconditional_flag_settles_it_again(tmp_path):
    root = _checkout(tmp_path)
    path = _write(root, "test_settled.py",
                  "FLAG = True\n"
                  "class TestCond:\n"
                  "    if FLAG:\n        __test__ = False\n"
                  "    __test__ = True\n"
                  "\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["TestCond::test_softmax"]


def test_a_module_level_flag_follows_the_same_rules(tmp_path):
    """A later `__test__ = True` on the module re-enables it; a conditional one makes
    every node in it a maybe (pytest may collect nothing)."""
    root = _checkout(tmp_path)
    on = _write(root, "test_mod_on.py",
                "__test__ = False\n"
                "__test__ = True\n"
                "def test_a():\n    pass\n")
    assert accuracy_hints.test_nodes(on) == ["test_a"]
    cond = _write(root, "test_mod_cond.py",
                  "if True:\n    __test__ = False\n"
                  "def test_a():\n    pass\n")
    assert accuracy_hints.test_nodes(cond) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(cond)] == [
        ("test_a", False)]


def test_a_later_import_takes_the_testcase_identity_back(tmp_path):
    """`TC` is rebound by a second import, so it is no longer a TestCase: the class
    descends from something we cannot name and stays a maybe, not a target."""
    root = _checkout(tmp_path)
    path = _write(root, "test_rebind.py",
                  "from unittest import TestCase as TC\n"
                  "from builtins import object as TC\n"
                  "\n"
                  "class Helper(TC):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Helper::test_softmax", False), ("test_softmax_accuracy", True)]


def test_a_later_module_import_takes_the_alias_back(tmp_path):
    """`ut.TestCase` means unittest.TestCase only while `ut` is unittest."""
    root = _checkout(tmp_path)
    path = _write(root, "test_rebind_module.py",
                  "import unittest as ut\n"
                  "import os as ut\n"
                  "\n"
                  "class Custom(ut.TestCase):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []


def test_a_star_import_drops_the_certainties_it_may_shadow(tmp_path):
    """A star import can re-export the alias under a new value (`TC = object`), and its
    exports cannot be listed: every name bound so far becomes unknown, so the class is a
    maybe rather than a target."""
    root = _checkout(tmp_path)
    path = _write(root, "test_star.py",
                  "from unittest import TestCase as TC\n"
                  "from support_base import *\n"
                  "\n"
                  "class Helper(TC):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Helper::test_softmax", False), ("test_softmax_accuracy", True)]


def test_an_explicit_import_after_a_star_import_rebuilds_the_certainty(tmp_path):
    root = _checkout(tmp_path)
    path = _write(root, "test_star_then.py",
                  "from support_base import *\n"
                  "from unittest import TestCase as TC\n"
                  "\n"
                  "class Custom(TC):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["Custom::test_softmax"]


def test_a_star_import_disarms_a_module_alias(tmp_path):
    """`ut.TestCase` is only unittest's while `ut` still is: a star import may rebind it."""
    root = _checkout(tmp_path)
    path = _write(root, "test_star_alias.py",
                  "import unittest as ut\n"
                  "from support_base import *\n"
                  "\n"
                  "class Custom(ut.TestCase):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []


def test_a_diamond_resolves_the_flag_along_the_mro(tmp_path):
    """`TestSoftmax(Left, Right)`: Left inherits True from Root but does not define it,
    so the MRO reaches Right's False first — pytest skips the class."""
    root = _checkout(tmp_path)
    path = _write(root, "test_diamond.py",
                  "class Root:\n    __test__ = True\n"
                  "\n"
                  "class Left(Root):\n    pass\n"
                  "\n"
                  "class Right(Root):\n    __test__ = False\n"
                  "\n"
                  "class TestSoftmax(Left, Right):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_a_subclass_flag_still_beats_the_mro(tmp_path):
    root = _checkout(tmp_path)
    path = _write(root, "test_diamond_on.py",
                  "class Root:\n    __test__ = True\n"
                  "class Left(Root):\n    pass\n"
                  "class Right(Root):\n    __test__ = False\n"
                  "class TestSoftmax(Left, Right):\n"
                  "    __test__ = True\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["TestSoftmax::test_softmax"]


def test_a_normal_multiple_inheritance_is_collected(tmp_path):
    """`class Custom(unittest.TestCase, Mixin)` is collected: no class on its MRO sets
    `__test__`, and a TestCase base ignores the constructor rule."""
    root = _checkout(tmp_path)
    path = _write(root, "test_multi.py",
                  "import unittest\n"
                  "class Mixin:\n    def helper(self):\n        pass\n"
                  "class Custom(unittest.TestCase, Mixin):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["Custom::test_softmax"]


def test_an_inconsistent_hierarchy_is_uncertain(tmp_path):
    """`class Z(X, Y)` with `Y(X)` is a hierarchy Python itself rejects; the MRO cannot be
    built, so the class stays a maybe instead of a target."""
    root = _checkout(tmp_path)
    path = _write(root, "test_inconsistent.py",
                  "class X:\n    pass\n"
                  "class Y(X):\n    pass\n"
                  "class Z(X, Y):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Z::test_softmax", False)]


def test_a_redefined_ancestor_does_not_rewrite_inheritance(tmp_path):
    """`Middle` keeps the `Root` it saw at its own definition; redefining `Root` later
    does not change what `Middle` inherits, so `__test__` stays the first Root's False."""
    root = _checkout(tmp_path)
    path = _write(root, "test_rebind_base.py",
                  "class Root:\n    __test__ = False\n"
                  "\n"
                  "class Middle(Root):\n    pass\n"
                  "\n"
                  "class Root:\n    __test__ = True\n"
                  "\n"
                  "class TestSoftmax(Middle):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [n.node for n in accuracy_hints.test_node_info(path)] == ["test_softmax_accuracy"]


def test_a_redefined_ancestor_does_not_disable_a_live_class_either(tmp_path):
    """The mirror image: the first `Root` says True, the later one says False, and the
    subclass still inherits True."""
    root = _checkout(tmp_path)
    path = _write(root, "test_rebind_base_on.py",
                  "class Root:\n    __test__ = True\n"
                  "class Middle(Root):\n    pass\n"
                  "class Root:\n    __test__ = False\n"
                  "class TestSoftmax(Middle):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == ["TestSoftmax::test_softmax"]


def test_a_dynamic_base_is_uncertain_not_ignored(tmp_path):
    """`class TestSoftmax(make_base())` has a base we cannot name; dropping the expression
    would read as "no bases", which collects a class pytest skips (`Base.__test__`)."""
    root = _checkout(tmp_path)
    path = _write(root, "test_dyn_base.py",
                  "class Base:\n    __test__ = False\n"
                  "\n"
                  "def make_base():\n    return Base\n"
                  "\n"
                  "class TestSoftmax(make_base()):\n"
                  "    def test_softmax(self):\n        pass\n"
                  "\n"
                  "def test_softmax_accuracy():\n    pass\n")
    assert accuracy_hints.test_nodes(path) == ["test_softmax_accuracy"]
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("TestSoftmax::test_softmax", False), ("test_softmax_accuracy", True)]


def test_a_subscripted_base_is_uncertain_too(tmp_path):
    """`Generic[T]` is not a name either: the class stays a maybe."""
    root = _checkout(tmp_path)
    path = _write(root, "test_generic.py",
                  "from typing import Generic, TypeVar\n"
                  "T = TypeVar('T')\n"
                  "class Custom(Generic[T]):\n"
                  "    def test_softmax(self):\n        pass\n")
    assert accuracy_hints.test_nodes(path) == []
    assert [(n.node, n.certain) for n in accuracy_hints.test_node_info(path)] == [
        ("Custom::test_softmax", False)]


def test_an_empty_same_name_file_does_not_hide_a_real_candidate(tmp_path):
    """A stale `tests/test_softmax.py` with no cases is a diagnostic, not a candidate: it
    must not stop the search for `test_activation.py::test_softmax_accuracy`."""
    root = _checkout(tmp_path)
    _write(root, "test_softmax.py", "")          # empty placeholder
    _write(root, "test_activation.py", "def test_softmax_accuracy():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "softmax", "softmax")
    assert [h.target for h in hints] == [
        "tests/test_activation.py::test_softmax_accuracy",   # a real node first
        "tests/test_softmax.py"]                             # the file as a fallback
    assert "整文件 target 交给 pytest 自己收集" in hints[-1].why
    assert "收集不到" not in hints[-1].why


def test_a_name_similar_file_without_related_cases_does_not_stop_the_search(tmp_path):
    """A file whose name matches but holds no related case is a fallback too: a named
    node in another file still wins and comes first."""
    root = _checkout(tmp_path)
    _write(root, "test_softmax_ops.py", "def test_unrelated():\n    pass\n")
    _write(root, "test_activation.py", "def test_softmax_accuracy():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "softmax", "softmax_other")
    assert hints[0].target == "tests/test_activation.py::test_softmax_accuracy"
    assert "tests/test_softmax_ops.py" in {h.target for h in hints}


def test_suggest_does_not_match_inside_a_longer_word(tmp_path):
    """A bare substring hit put `mul` inside `cumulative`: a key only counts on a `_`
    boundary, or every `mul`-ish op inherits every `accumulate`-ish test."""
    root = _checkout(tmp_path)
    _write(root, "test_other.py", "def test_cumulative_sum():\n    pass\n")
    assert accuracy_hints.suggest(_index(root), "mul", "mul") == []


def test_suggest_prefers_the_ops_own_name_over_its_family_file(tmp_path):
    """`mhc_post` sits in `benchmark/test_mhc.py`; matching the family name `mhc` against
    function names pulled in `test_mhc_bwd_vs_ref`, a different op."""
    root = _checkout(tmp_path)
    _write(root, "test_mhc_ops.py", "def test_mhc_bwd_vs_ref():\n    pass\n"
                                    "def test_mhc_post_vs_ref():\n    pass\n")
    hints = accuracy_hints.suggest(_index(root), "mhc_post", "mhc",
                                  names=["mhc_post"])
    assert "tests/test_mhc_ops.py::test_mhc_post_vs_ref" in {h.target for h in hints}
    assert "tests/test_mhc_ops.py::test_mhc_bwd_vs_ref" not in {h.target for h in hints}


def test_suggest_without_a_tests_dir_is_empty(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert accuracy_hints.suggest(_index(root), "kda", "kda") == []


def _entry(run="main/kda", repo="main", op="kda", op_file="kda", sides=None, hints=()):
    return accuracy_hints.Entry(run, repo, op, op_file, f"benchmark/test_{op_file}.py",
                                sides or {"a": "no_test", "b": "no_test"}, list(hints))


def test_write_reports_lists_the_ops_and_keeps_the_draft_commented(tmp_path):
    """The draft must be usable: every candidate line commented, so a wrong guess never
    reaches --accuracy-map by accident."""
    hint = accuracy_hints.Hint(1, "同名文件", "tests/test_kda.py::test_kda_gate",
                               "同名文件存在，但没有 `test_kda`")
    entries = [_entry(hints=[hint]), _entry(run="vllm/empty", repo="vllm",
                                            op="empty", op_file="empty")]
    md_path, map_path = accuracy_hints.write_reports(tmp_path / "out", entries)
    md = md_path.read_text(encoding="utf-8")
    assert md.startswith("# 精度门禁：未找到单测的算子")
    assert "共 2 个算子（main 1、vllm 1）：1 个找到候选，1 个没有。" in md
    assert "`main:kda@kda`" in md and "`tests/test_kda.py::test_kda_gate`" in md
    assert "## 没有候选" in md and "`vllm:empty@empty`" in md

    draft = map_path.read_text(encoding="utf-8")
    body = [line for line in draft.splitlines() if line and not line.startswith("#")]
    assert body == []                                   # nothing to paste by accident
    assert "# main:kda@kda tests/test_kda.py::test_kda_gate" in draft
    assert "每个算子只留一行" in draft


def test_write_from_gate_writes_both_files_and_returns_the_count(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    _write(root, "test_kda.py", "def test_kda_gate():\n    pass\n")
    monkeypatch.setitem(accuracy_hints.REPOS, "main", root)
    monkeypatch.setattr(accuracy_hints.run_batch, "load_inventory", lambda: {"main": []})
    case = run_batch.Case("main", "kda", "kda", "kda")
    passing = {"a": {"main/kda": {"status": "pass"}}, "b": {"main/kda": {"status": "pass"}}}
    md_path, map_path, count = accuracy_hints.write_from_gate(tmp_path / "out", [case],
                                                              passing)
    assert count == 0 and md_path.is_file() and map_path.is_file()
    assert "当前记录里没有 `no_test` 算子" in md_path.read_text(encoding="utf-8")

    gate = {"a": {"main/kda": {"status": "no_test"}}, "b": {"main/kda": {"status": "pass"}}}
    md_path, map_path, count = accuracy_hints.write_from_gate(tmp_path / "out", [case],
                                                              gate)
    assert count == 1
    md = md_path.read_text(encoding="utf-8")
    assert "`main:kda@kda`" in md and "a `no_test`，b `pass`" in md
    assert "tests/test_kda.py::test_kda_gate" in md


def test_write_from_gate_keeps_the_ops_this_run_did_not_select(tmp_path, monkeypatch):
    """The gate keeps every run it ever recorded for this --out; a resume that re-ran one
    op must not rewrite the report as if the others had never been there."""
    root = _checkout(tmp_path)
    _write(root, "test_alpha.py", "def test_alpha_accuracy():\n    pass\n")
    _write(root, "test_beta.py", "def test_beta_accuracy():\n    pass\n")
    monkeypatch.setitem(accuracy_hints.REPOS, "main", root)
    monkeypatch.setattr(accuracy_hints.run_batch, "load_inventory", lambda: {"main": []})
    gate = {side: {"main/alpha": {"status": "no_test"},
                   "main/beta": {"status": "no_test"}} for side in ("a", "b")}
    md_path, _, count = accuracy_hints.write_from_gate(
        tmp_path / "out", [run_batch.Case("main", "alpha", "alpha", "alpha")], gate)
    text = md_path.read_text(encoding="utf-8")
    assert count == 2
    assert "`main:alpha@alpha`" in text and "`main:beta@beta`" in text


def test_write_from_gate_empties_the_list_when_nothing_is_left(tmp_path, monkeypatch):
    """A run that fixed every op has to say so: leaving the old list in place would keep
    pointing at ops that no longer need an --accuracy-map entry."""
    root = _checkout(tmp_path)
    _write(root, "test_kda.py", "def test_kda_gate():\n    pass\n")
    monkeypatch.setitem(accuracy_hints.REPOS, "main", root)
    monkeypatch.setattr(accuracy_hints.run_batch, "load_inventory", lambda: {"main": []})
    case = run_batch.Case("main", "kda", "kda", "kda")
    out = tmp_path / "out"
    no_test = {side: {"main/kda": {"status": "no_test"}} for side in ("a", "b")}
    accuracy_hints.write_from_gate(out, [case], no_test)
    assert "`main:kda@kda`" in (out / accuracy_hints.MD_NAME).read_text(encoding="utf-8")

    fixed = {side: {"main/kda": {"status": "pass"}} for side in ("a", "b")}
    md_path, map_path, count = accuracy_hints.write_from_gate(out, [case], fixed)
    assert count == 0
    md = md_path.read_text(encoding="utf-8")
    assert "`main:kda@kda`" not in md
    assert "当前记录里没有 `no_test` 算子" in md
    assert "这份清单是空的" in map_path.read_text(encoding="utf-8")


def test_build_entries_reads_the_no_test_runs_of_an_ab_json(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    _write(root, "test_kda.py", "def test_kda_gate():\n    pass\n")
    monkeypatch.setitem(accuracy_hints.REPOS, "main", root)
    monkeypatch.setattr(accuracy_hints.run_batch, "load_inventory", lambda: {"main": []})
    out = tmp_path / "ab"
    out.mkdir()
    (out / "ab.json").write_text(json.dumps({"accuracy": {"sides": {
        "a": {"main/kda": {"status": "no_test"}},
        "b": {"main/kda": {"status": "pass"},
              "main/other": {"status": "pass"}}}}}))
    entries = accuracy_hints.build_entries(out)
    assert [e.run for e in entries] == ["main/kda"]
    assert entries[0].op == "kda" and entries[0].op_file == "kda"
    assert [h.target for h in entries[0].hints] == ["tests/test_kda.py::test_kda_gate"]


def test_main_writes_the_files_for_an_existing_ab_dir(tmp_path, monkeypatch, capsys):
    root = _checkout(tmp_path)
    _write(root, "test_kda.py", "def test_kda_gate():\n    pass\n")
    monkeypatch.setitem(accuracy_hints.REPOS, "main", root)
    monkeypatch.setattr(accuracy_hints.run_batch, "load_inventory", lambda: {"main": []})
    out = tmp_path / "ab"
    out.mkdir()
    (out / "ab.json").write_text(json.dumps({"accuracy": {"sides": {
        "a": {"main/kda": {"status": "no_test"}}}}}))
    monkeypatch.setattr("sys.argv", ["accuracy_hints.py", "--ab-dir", str(out)])
    accuracy_hints.main()
    assert (out / accuracy_hints.MD_NAME).is_file()
    assert (out / accuracy_hints.MAP_NAME).is_file()
    assert "1 ops without a test" in capsys.readouterr().out


def test_main_refuses_a_dir_without_ab_json(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["accuracy_hints.py", "--ab-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="no .*ab.json"):
        accuracy_hints.main()


def test_ab_run_writes_the_hint_files_after_the_gate(tmp_path, monkeypatch):
    """`--accuracy-gate` ends with the two files, and a failing hint report must not take
    the run down with it."""
    root = _checkout(tmp_path)
    _write(root, "test_softmax.py", "def test_softmax_accuracy():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    args = argparse.Namespace(
        out=tmp_path / "ab", accuracy_timeout=60, use_flagtune=0,
        accuracy_map={}, accuracy_map_file=None,
        sides={"a": {"name": "A", "pythonpath": "", "env": {}},
               "b": {"name": "B", "pythonpath": "", "env": {}}})
    args.out.mkdir()
    monkeypatch.setattr(ab_run, "run_accuracy",
                        lambda *a: {"status": ab_run.GATE_NO_TEST, "target": "",
                                    "return_code": "0", "elapsed_ms": "0", "log": "",
                                    "detail": ""})
    ab_run.accuracy_gate([run_batch.Case("main", "softmax", "softmax", "softmax")],
                         args, ["0"])
    assert (args.out / "accuracy_no_test.md").is_file()
    assert (args.out / "accuracy_map.suggested.txt").is_file()
