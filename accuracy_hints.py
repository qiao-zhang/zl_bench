#!/usr/bin/env python3
"""Suggest the unit test of the ops ab_run.py --accuracy-gate could not locate.

`no_test` is not a verdict: the gate found no test to run, so the op's A/B stands
unverified. The benchmark inventory and the upstream tests are not 1:1 (`perf_`
prefixes, inplace/out variants sharing one test, names that have nothing to do with
the test function), and finding the missing ones means searching two checkouts. This
module does that search and writes what it found next to the A/B output:

  accuracy_no_test.md          what the gate could not test, and where to look
  accuracy_map.suggested.txt   the same candidates in --accuracy-map form, `###` prose
                               and one `#` per candidate line (delete that `#` to use it)

A candidate is a `tests/...py[::test_func]` target, tagged by how it was matched:

  [同名文件]      tests/test_<op_file>.py exists but has no test_<op>: its own cases
  [文件名相近]    another file whose name normalises to the op or one of its op_names
  [文件名包含]    a file name that contains the op (or the op contains it)
  [函数名命中]    a function whose name carries the op, in a file named otherwise
  [benchmark 名命中]  a function named after the benchmark file (`test_sparse_attention`
                  for `sparse_attn_triton`): the weakest match, tried last
  [整文件]        a name-similar file with no related function; the whole file gates it

None of it is a decision: a wrong target fails the op, or gates it with a test that
does not cover it -- which is why every line says how it was matched and the draft
keeps every candidate commented, one per op to uncomment. Prose carries `###`, so
only a candidate line becomes an entry when its single `#` is deleted.

The function-name sweep parses every test file of the repo, so it only runs for the
ops the file names say nothing about.

  python accuracy_hints.py --ab-dir runs/ab_x       # rewrite both files
  ab_run.py ... --accuracy-gate                     # writes them automatically
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import run_batch
from ab_compare import MANIFEST
from gen_inventory import REPOS

LOG = "[accuracy-hints]"
TESTS = "tests"
MD_NAME = "accuracy_no_test.md"
MAP_NAME = "accuracy_map.suggested.txt"
NO_TEST = "no_test"          # ab_run.GATE_NO_TEST
MAX_FILES = 6                # name-similar files listed per op
MAX_HINTS = 12               # candidates listed per op (report and draft)
MAX_MAP = 3                  # candidates copied into the editable draft per op
MIN_KEY = 3                  # shorter normalised names match everything
_PREFIX = ("test_", "perf_")


def normalize(name: str) -> str:
    """`test_FLA/test_chunk_gdn2.py` / `perf_chunk_gdn2` -> `chunk_gdn2`.

    Only the decorations the two repos use for "the same op, another entry point":
    a `test_`/`perf_` prefix and a `_perf` suffix. `_triton` / `_ascend` are kept --
    they name different kernels, and matching them would suggest the wrong test."""
    n = Path(name).name.lower()
    if n.endswith(".py"):
        n = n[:-3]
    changed = True
    while changed:
        changed = False
        for pre in _PREFIX:
            if n.startswith(pre):
                n = n[len(pre):]
                changed = True
    for suf in ("_perf", "_benchmark"):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return n


class Hint(NamedTuple):
    tier: int      # 1 同名文件 .. 5 benchmark 名命中
    tag: str       # the bracket in the report and the draft
    target: str    # tests/...py or tests/...py::test_func
    why: str       # how it was matched, in the words the reader needs
    certain: bool = True   # False: defined under if/try, so it may not exist


def _keys(op: str, op_file: str, names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(file keys, function keys).

    The benchmark file name is a file key only: `test_mhc.py` says `mhc` while the op is
    `mhc_post`, and matching that against function names pulled in `test_mhc_bwd_vs_ref`
    for `mhc_post` — a different op in the same family."""
    def uniq(raw_names: Sequence[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for raw in raw_names:
            key = normalize(raw)
            if len(key) >= MIN_KEY:
                seen.setdefault(key, None)
        return list(seen)

    return uniq([op, op_file, *names]), uniq([op, *names])


def _patterns(keys: Sequence[str]) -> List[Tuple[str, "re.Pattern"]]:
    """Compile each key as a word inside a normalised name.

    A bare substring hit matched `mul` inside `cumulative` and `mhc` inside
    `mhc_bwd_vs_ref`; the boundary keeps `test_mul_broadcast_shape` and drops the rest."""
    return [(key, re.compile(rf"(^|_){re.escape(key)}(_|$)")) for key in keys]


def _hit(norm: str, patterns: Sequence[Tuple[str, "re.Pattern"]]) -> Optional[str]:
    """The key this normalised name carries (on a `_` boundary), or None."""
    for key, pattern in patterns:
        if pattern.search(norm):
            return key
    return None


class TestsIndex:
    """One repo's tests/ tree: its test files, and their test functions on demand."""

    def __init__(self, root: Path):
        self.root = root
        self.tests = root / TESTS
        self.exists = self.tests.is_dir()
        self.rels = ([p.relative_to(self.tests).as_posix()
                      for p in sorted(self.tests.rglob("test_*.py"))]
                     if self.exists else [])
        self._funcs: Dict[str, List[TestNode]] = {}
        self._all: Optional[List[Tuple[str, TestNode, str]]] = None

    def funcs(self, rel: str) -> List[TestNode]:
        """The nodes a file defines, in file order, each with its certainty."""
        if rel not in self._funcs:
            self._funcs[rel] = test_node_info(self.tests / rel)
        return self._funcs[rel]

    def all_funcs(self) -> List[Tuple[str, TestNode, str]]:
        """(rel, node, its normalised function name) over the whole tree, parsed and
        normalised once: the sweep for a file-name mismatch visits thousands of names."""
        if self._all is None:
            self._all = [(rel, n, normalize(node_leaf(n.node)))
                         for rel in self.rels for n in self.funcs(rel)]
        return self._all


class TestNode(NamedTuple):
    node: str        # `test_x` / `TestA::test_x`
    certain: bool    # defined unconditionally; a `def` under if/try may not exist


def _dotted(expr: ast.expr) -> str:
    """`unittest.TestCase` / `TestCase` / `ut.TestCase` -> the dotted source name."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        head = _dotted(expr.value)
        return f"{head}.{expr.attr}" if head else expr.attr
    return ""


def _effective_test_flag(body: Sequence[ast.stmt]) -> str:
    """The `__test__` value pytest will see for a module or class body, in order.

    "true" / "false" / "unset" (never assigned) / "unknown". A later assignment wins
    (`__test__ = True` then `= False` ends False), an assignment under `if`/`try`/...
    may or may not have run so it leaves the value unknown, and a later unconditional
    assignment settles it again. A `with` body always runs. Class and function bodies
    are their own scope and are skipped."""
    flag = "unset"

    def scan(statements: Sequence[ast.stmt], certain: bool) -> None:
        nonlocal flag
        for stmt in statements:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt.target, ast.Name):
                    targets = [stmt.target.id]
                else:
                    targets = []
                if "__test__" not in targets:
                    continue
                value = stmt.value
                if not certain:
                    flag = "unknown"
                elif isinstance(value, ast.Constant) and isinstance(value.value, bool):
                    flag = "true" if value.value else "false"
                else:
                    flag = "unknown"
                continue
            for sub, keeps in _sub_bodies(stmt):
                scan(sub, certain and keeps)

    scan(body, True)
    return flag


class ClassInfo(NamedTuple):
    """What a class definition is, as far as a subclass cares.

    `testcase` is the class's identity (a unittest.TestCase subclass, or not, or
    unknown); `ctor` says it carries a custom constructor (`__init__` or `__new__`,
    defined here or inherited); `test_flag` is the effective `__test__` pytest will see;
    `own_flag` is what this class itself assigns (a subclass inherits the effective one
    through the MRO); `bases` holds the base classes *as they were bound at this
    definition*, not their names, because rebinding a name later (`class Root:` defined
    twice) does not change what an existing class inherits from. `unresolved` marks a
    base we could not identify. None of it says whether the class itself is collectable:
    that is decided per class."""
    testcase: Optional[bool]
    ctor: Optional[bool]
    test_flag: str = "unset"       # effective value, resolved along the MRO
    own_flag: str = "unset"        # this class's own `__test__`, for its subclasses
    unresolved: bool = False       # a base we cannot name
    bases: Tuple["ClassInfo", ...] = ()

    @staticmethod
    def unknown() -> "ClassInfo":
        return ClassInfo(None, None, "unknown", "unknown")

    @staticmethod
    def a_module() -> "ClassInfo":
        """A module name (`import unittest`): not a TestCase class, no constructor."""
        return ClassInfo(False, False)


# `unittest.TestCase` itself defines no `__test__`: an MRO leaf.
_TESTCASE_LEAF = ClassInfo(True, False)


def _defines_ctor(node: ast.ClassDef) -> bool:
    """Whether the class body defines `__init__` or `__new__` itself."""
    return any(isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and
               stmt.name in ("__init__", "__new__") for stmt in node.body)


def _or_unknown(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
    """Three-valued or: True wins, an unknown side keeps the result unknown."""
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


def _merge(seqs: List[List[ClassInfo]]) -> Optional[List[ClassInfo]]:
    """C3 linearisation; None when the hierarchy is inconsistent.

    Classes are compared by identity: two definitions that happen to look alike (`class
    Root: pass` written twice) are different bases."""
    out: List[ClassInfo] = []
    work = [list(seq) for seq in seqs if seq]
    while work:
        for seq in work:
            head = seq[0]
            if not any(head is other for s in work for other in s[1:]):
                break
        else:
            return None
        out.append(head)
        work = [s[1:] if s[0] is head else s for s in work]
        work = [s for s in work if s]
    return out


def _mro(info: ClassInfo) -> Optional[List[ClassInfo]]:
    if info.unresolved:
        return None
    tail = _mro_after(info.bases)
    return None if tail is None else [info] + tail


def _mro_after(bases: Sequence[ClassInfo]) -> Optional[List[ClassInfo]]:
    """The MRO of a list of bases (the subclass's own entry excluded)."""
    parts: List[List[ClassInfo]] = []
    for base in bases:
        sub = _mro(base)
        if sub is None:
            return None
        parts.append(sub)
    parts.append(list(bases))
    return _merge(parts)


def _class_info(node: ast.ClassDef, attrs: Dict[str, Optional[bool]],
                names: Dict[str, ClassInfo]) -> ClassInfo:
    """The identity this class passes on to its subclasses.

    The constructor flag is inherited from the bases this module defines: pytest skips
    `class TestSoftmax(Base)` when `Base` brings an `__init__`, and a subclass of it
    inherits that. The `__test__` flag follows Python's MRO — pytest reads it with
    `getattr`, so the first class in the MRO that defines it decides (a base's effective
    value is not what a diamond returns), and a class that sets its own wins. The bases
    are captured as the classes bound at this statement, so a later redefinition of an
    ancestor's name does not rewrite this class's inheritance. A base we cannot identify
    (`from .base import BaseTestCase`) or an inconsistent hierarchy leaves it unknown."""
    own = _effective_test_flag(node.body)
    ctor: Optional[bool] = _defines_ctor(node)
    bases: List[ClassInfo] = []
    unresolved = False
    for expr in node.bases:
        base = _dotted(expr)
        if not base:
            # `make_base()`, `Generic[T]`, ...: an expression we cannot identify. Dropping
            # it would read as "no bases", which is what collects a class pytest skips.
            unresolved = True
            continue
        if base in attrs:
            if attrs[base] is None:
                unresolved = True
            else:
                bases.append(_TESTCASE_LEAF)
        elif base in names:
            bases.append(names[base])
        else:
            unresolved = True                    # imported from somewhere we cannot see
    if unresolved:
        return ClassInfo(None, None, "unknown", own, True, tuple(bases))
    for base in bases:
        ctor = _or_unknown(ctor, base.ctor)
    identities = [base.testcase for base in bases]
    if any(state is True for state in identities):
        testcase: Optional[bool] = True
    elif any(state is None for state in identities):
        testcase = None
    else:
        testcase = False
    flag = own
    if flag == "unset":
        order = _mro_after(bases)
        if order is None:
            flag = "unknown"
        else:
            for base in order:
                if base.own_flag != "unset":
                    flag = base.own_flag
                    break
    return ClassInfo(testcase, ctor, flag, own, False, tuple(bases))


def _class_state(node: ast.ClassDef, attrs: Dict[str, Optional[bool]],
                 names: Dict[str, ClassInfo]) -> Optional[bool]:
    """Whether pytest collects this class: True, False, or None when it cannot tell.

    `__test__ = False` — its own or one it inherits — makes pytest skip the class
    whatever else it is, a `unittest.TestCase` subclass included; an unreadable value
    leaves it uncertain. A TestCase subclass follows unittest's rules (its own
    `__init__` is normal there). Any other class is an ordinary pytest class: pytest
    takes it when its name carries the `Test*` prefix **or** it sets `__test__ = True`,
    and skips it when it has a custom constructor (own or inherited). A class whose base
    cannot be resolved may or may not be a TestCase, so it stays uncertain."""
    info = _class_info(node, attrs, names)
    if info.test_flag == "false":
        return False
    if info.test_flag == "unknown":
        return None
    if info.testcase is True:
        return True                              # unittest's rules: __init__ is fine
    if info.testcase is None or info.ctor is None:
        return None                              # maybe a TestCase, maybe not
    if not (node.name.startswith("Test") or info.test_flag == "true"):
        return False                             # neither a Test* name nor __test__ = True
    return False if info.ctor else True


def _sub_bodies(node: ast.stmt) -> List[Tuple[Sequence[ast.stmt], bool]]:
    """(statements, still-certain) for the bodies a compound statement holds.

    A `with` body always runs; an `if`/`try`/`for`/`while`/`match` body may not, so a
    `def` inside it is a node pytest only sometimes has."""
    out: List[Tuple[Sequence[ast.stmt], bool]] = []
    for field in ("body", "orelse", "finalbody"):
        body = getattr(node, field, None)
        if isinstance(body, list):
            out.append((body, isinstance(node, (ast.With, ast.AsyncWith))))
    for handler in getattr(node, "handlers", []) or []:
        out.append((handler.body, False))
    for case in getattr(node, "cases", []) or []:
        out.append((case.body, False))
    return out


def test_node_info(path: Path) -> List[TestNode]:
    """The pytest nodes a module defines, with whether each is certain to exist.

    `test_x` at module level, `TestA::test_x` inside a collected class (every nesting
    level kept), and defs under an `if`/`try` marked as uncertain: pytest collects them
    only when their branch ran, which a static read cannot decide. The module scope's
    bindings are followed in definition order, so a base class means what it meant at
    the `class` statement: a local import inside a function binds nothing at module
    level, a later `class Base` / `Base = ...` takes an earlier TestCase binding back,
    and a conditional import leaves that name unknown. A function nested in a function
    is never collected at all."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return []
    module_flag = _effective_test_flag(tree.body)   # `__test__` on the module itself
    if module_flag == "false":                      # pytest collects nothing from it
        return []
    # Module-scope bindings: `ut.TestCase` -> is it unittest.TestCase, and bare name ->
    # same question. None means "bound, but we cannot tell what to" (a conditional
    # import, a name from elsewhere).
    attrs: Dict[str, Optional[bool]] = {}
    names: Dict[str, ClassInfo] = {}
    out: List[TestNode] = []

    def take(name: str, info: ClassInfo, certain: bool) -> None:
        """Rebind a module-scope name; a conditional binding leaves it unknown."""
        names[name] = info if certain else ClassInfo.unknown()
        for key in [k for k in attrs if k.split(".")[0] == name]:
            attrs[key] = None            # `unittest = ...` kills `unittest.TestCase`

    def bind_import(node: ast.stmt, certain: bool) -> bool:
        """Rebind every name an import binds, not just the unittest ones.

        A later import of the same name takes the TestCase identity back:
        `from unittest import TestCase as TC` followed by `from builtins import object
        as TC` leaves `TC` as object, so a subclass of it is no test. What another
        module exports we cannot know, so those names become unknown rather than keeping
        the old identity."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("unittest", "unittest.case"):
                    module = alias.asname or alias.name
                    take(module, ClassInfo.a_module(), certain)
                    attrs[f"{module}.TestCase"] = True if certain else None
                else:
                    # `import a.b` binds `a`; a module is not a TestCase class.
                    bound = alias.asname or alias.name.split(".")[0]
                    take(bound, ClassInfo.a_module(), certain)
            return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            unittest_source = module in ("unittest", "unittest.case")
            for alias in node.names:
                if alias.name == "*":
                    if unittest_source:
                        take("TestCase", ClassInfo(True, False), certain)
                    else:
                        # The module's exports cannot be listed, so every name bound so
                        # far may have been shadowed -- a helper that re-exports
                        # `TestCase` under the old alias does exactly that. Drop the
                        # certainties instead of keeping them; a later explicit import or
                        # class definition rebuilds the one we can see.
                        for name in list(names):
                            names[name] = ClassInfo.unknown()
                        for key in list(attrs):
                            attrs[key] = None
                    continue
                bound = alias.asname or alias.name
                if unittest_source and alias.name == "TestCase":
                    take(bound, ClassInfo(True, False), certain)
                else:
                    take(bound, ClassInfo.unknown(), certain)
            return True
        return False

    def bind_assign(node: ast.stmt, certain: bool) -> bool:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            return False
        for name in targets:
            # `Base = something` says nothing about what: a subclass of it stays a
            # maybe, rather than being judged as an ordinary class.
            take(name, ClassInfo.unknown(), certain)
        return True

    def walk(body: Sequence[ast.stmt], prefix: str, certain: bool,
             module_scope: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    out.append(TestNode(prefix + node.name, certain))
            elif isinstance(node, ast.ClassDef):
                state = _class_state(node, attrs, names)
                if module_scope:
                    # Only the identity and the constructor flag propagate: a collectable
                    # `TestBase` is not a TestCase, so `class Helper(TestBase)` must be
                    # judged on its own name and on the `__init__` it inherits.
                    take(node.name, _class_info(node, attrs, names), certain)
                if state is not False:
                    # A class body is its own scope: what it imports or assigns does not
                    # bind at module level.
                    walk(node.body, f"{prefix}{node.name}::", certain and state is True,
                         False)
            elif module_scope and bind_import(node, certain):
                pass
            elif module_scope and bind_assign(node, certain):
                pass
            else:
                for sub, keeps in _sub_bodies(node):
                    walk(sub, prefix, certain and keeps, module_scope)

    walk(tree.body, "", module_flag != "unknown", True)
    return out


def test_nodes(path: Path) -> List[str]:
    """The nodes pytest certainly collects (the uncertain ones are reported separately)."""
    return [n.node for n in test_node_info(path) if n.certain]


def node_leaf(node: str) -> str:
    """`TestA::test_x` -> `test_x`: matching and similarity are about the function."""
    return node.rsplit("::", 1)[-1]


def _dir_rank(rel: str, same: str) -> int:
    """Rank candidate files: the op's own tests/ subdir first, framework dirs last."""
    here = str(Path(rel).parent).replace("\\", "/")
    want = str(Path(same).parent).replace("\\", "/")
    if here == want:
        return 0
    if here in (".", ""):
        return 1
    if here.split("/")[0].startswith("test_"):
        return 2
    return 3


def _similarity(func: str, op: str) -> float:
    """How close a node's function name is to the op, for ordering a same-file list."""
    return difflib.SequenceMatcher(None, normalize(node_leaf(func)), normalize(op)).ratio()


def suggest(index: TestsIndex, op: str, op_file: str,
            names: Sequence[str] = ()) -> List[Hint]:
    """Ranked candidates for one op, from how its file and function names look.

    Within a file the certain nodes come first: a `def` under an `if`/`try` is only
    there when its branch ran, so it is listed with `certain=False` and the report marks
    it. ab_run.py's automatic lookup takes certain nodes only, because a node that is
    not there makes pytest exit 4 and drops the op as an error."""
    file_keys, func_keys = _keys(op, op_file, names)
    file_patterns, func_patterns = _patterns(file_keys), _patterns(func_keys)
    found: Dict[str, Hint] = {}
    # Whole-file targets are a fallback, not a candidate: they say "this file may hold the
    # test" without naming a case. They must not stop the search for a real node.
    deferred: List[Hint] = []

    def add(tier: int, tag: str, target: str, why: str, certain: bool = True) -> None:
        if target not in found:              # the first (most confident) tag wins
            found[target] = Hint(tier, tag, target, why, certain)

    def defer(tier: int, tag: str, target: str, why: str, certain: bool = True) -> None:
        deferred.append(Hint(tier, tag, target, why, certain))

    def order(nodes: Sequence[TestNode], by_name: bool) -> List[TestNode]:
        """Certain first, then (for a same-file list) by closeness to the op."""
        return sorted(nodes, key=lambda n: (not n.certain,
                                            -_similarity(n.node, op) if by_name else 0))

    same = f"test_{op_file}.py"              # what ab_run.auto_test_target looks for
    if same in index.rels:
        funcs = index.funcs(same)
        related = [n for n in funcs
                   if _hit(normalize(node_leaf(n.node)), func_patterns)]
        if not funcs:
            # An empty (or unparsable) file here does not mean the op has no test: the
            # candidate list is only static, and the real test may sit in another file.
            defer(1, "同名文件", f"{TESTS}/{same}",
                  "同名文件存在，但没有解析到 test 函数（可能是 import 失败或动态定义）："
                  "整文件 target 交给 pytest 自己收集")
        elif related:
            for n in order(related, by_name=False):
                add(1, "同名文件", f"{TESTS}/{same}::{n.node}",
                    f"同名文件里 `{n.node}` 的名字含 "
                    f"`{_hit(normalize(node_leaf(n.node)), func_patterns)}`，"
                    f"只是不叫 `test_{op}`", n.certain)
        else:
            for n in order(funcs, by_name=True):
                add(1, "同名文件", f"{TESTS}/{same}::{n.node}",
                    f"同名文件存在，但没有 `test_{op}`；这个文件有 {len(funcs)} 个用例，"
                    "按名字与算子的接近程度排序", n.certain)

    others = (rel for rel in index.rels if rel != same)
    candidates: List[Tuple[int, int, str, str]] = []
    for rel in others:
        carried = normalize(rel) if normalize(rel) in file_keys else None
        tier = 2 if carried else 3
        if carried is None:
            carried = _hit(normalize(rel), file_patterns)
        if carried is None:
            continue
        candidates.append((tier, _dir_rank(rel, same), rel, carried))
    # Rank every matching file first, cap afterwards: capping in directory order hid the
    # exact-name file behind six same-directory near misses.
    candidates.sort()
    for tier, _, rel, carried in candidates[:MAX_FILES]:
        tag = "文件名相近" if tier == 2 else "文件名包含"
        funcs = index.funcs(rel)
        related = [n for n in funcs
                   if _hit(normalize(node_leaf(n.node)), func_patterns)]
        if related:
            for n in order(related, by_name=False):
                add(tier, tag, f"{TESTS}/{rel}::{n.node}",
                    f"文件名里含 `{carried}`，`{n.node}` 这个名字含 "
                    f"`{_hit(normalize(node_leaf(n.node)), func_patterns)}`", n.certain)
        elif funcs:
            defer(tier, tag, f"{TESTS}/{rel}",
                  f"文件名与 `{carried}` 相近，但文件里没有名字相关的用例（共 {len(funcs)} 个）："
                  "整文件 target 会把它们都算作此算子的门禁")

    if not found:
        # Nothing in the file names: only then pay for parsing every test file, and say
        # which name hit -- the file may be about something else entirely.
        hits = [(rel, n) for rel, n, norm in index.all_funcs()
                if _hit(norm, func_patterns) is not None]
        hits.sort(key=lambda h: (not h[1].certain, h[0], h[1].node))
        for rel, n in hits:
            add(4, "函数名命中", f"{TESTS}/{rel}::{n.node}",
                f"`{rel}` 的 `{n.node}` 名字含 "
                f"`{_hit(normalize(node_leaf(n.node)), func_patterns)}`，但文件名与算子无关",
                n.certain)

    extra = _patterns([k for k in file_keys if k not in func_keys])
    if not found and extra:
        # The op's own name is nowhere, but the benchmark file it lives in may name the
        # test: sparse_attn_triton sits in test_sparse_attention.py. Weaker, and it can
        # pull in a sibling op of the same family, so it only runs when nothing else did.
        for rel, n, norm in index.all_funcs():
            carried = _hit(norm, extra)
            if carried is not None:
                add(5, "benchmark 名命中", f"{TESTS}/{rel}::{n.node}",
                    f"`{rel}` 的 `{n.node}` 含 benchmark 文件名 `{carried}`，"
                    f"但算子是 `{op}`：确认它是否覆盖这个算子", n.certain)

    # Whole-file targets come last: a named case is always the better answer.
    for hint in deferred:
        if hint.target not in found:
            found[hint.target] = hint
    # Insertion order already is the ranking (tier 1 first, then the sorted files, and
    # the same-file list in similarity order); re-sorting by target would throw that away.
    return list(found.values())[:MAX_HINTS]


class Entry(NamedTuple):
    run: str                       # <repo>/<run_name>, ab.json's key
    repo: str
    op: str
    op_file: str
    bench: str                     # its benchmark file, from the inventory
    sides: Dict[str, str]          # side -> gate status
    hints: List[Hint]


def _inventory_by_run() -> Dict[str, Dict[str, str]]:
    try:
        return run_batch.run_index(run_batch.load_inventory())
    except SystemExit:
        return {}


def _entry(run: str, repo: str, op: str, op_file: str, sides: Dict[str, str],
           row: Optional[Dict[str, str]], indexes: Dict[str, TestsIndex]) -> Entry:
    row = row or {}
    names = [n for n in (row.get("op_names") or "").split(";") if n]
    index = indexes.setdefault(repo, TestsIndex(REPOS.get(repo) or Path(TESTS)))
    return Entry(run, repo, op, op_file, row.get("file", ""), sides,
                 suggest(index, op, op_file, names))


def no_test_runs(manifest: dict) -> List[Tuple[str, Dict[str, str]]]:
    """(run, {side: status}) for the runs at least one side recorded as no_test."""
    sides = (manifest.get("accuracy") or {}).get("sides") or {}
    out = []
    for run in sorted({run for rows in sides.values() for run in rows}):
        statuses = {side: (sides.get(side) or {}).get(run, {}).get("status", "")
                    for side in ("a", "b")}
        if NO_TEST in statuses.values():
            out.append((run, statuses))
    return out


def build_entries(ab_dir: Path) -> List[Entry]:
    """The no_test ops of an ab.json, each with its ranked candidates."""
    manifest = json.loads((ab_dir / MANIFEST).read_text())
    rows = _inventory_by_run()
    indexes: Dict[str, TestsIndex] = {}
    out = []
    for run, statuses in no_test_runs(manifest):
        repo, _, rest = run.partition("/")
        row = rows.get(run) or {}
        op = row.get("op") or rest.split("@")[0]
        op_file = row.get("op_file") or op
        out.append(_entry(run, repo, op, op_file, statuses, row, indexes))
    return out


def entries_from_gate(cases, gate: Dict[str, Dict[str, Dict[str, str]]],
                      indexes: Optional[Dict[str, TestsIndex]] = None,
                      ) -> List[Entry]:
    """The no_test ops of a gate record, with their candidates.

    The gate holds every run it has ever recorded for this --out, not just the ones this
    invocation selected: a resume that re-ran one op must not drop the others from the
    report. Statuses come from the gate; repo/op/op_file come from the case when the run
    is one of this invocation's, else from the inventory, else from the run name."""
    rows = _inventory_by_run()
    indexes = {} if indexes is None else indexes
    by_run = {f"{c.repo}/{c.run_name}": c for c in cases}
    out = []
    for run in sorted({run for rows_ in gate.values() for run in rows_}):
        statuses = {side: (gate.get(side, {}).get(run) or {}).get("status", "")
                    for side in ("a", "b")}
        if NO_TEST not in statuses.values():
            continue
        case = by_run.get(run)
        row = rows.get(run) or {}
        repo, _, rest = run.partition("/")
        op = case.op if case else (row.get("op") or rest.split("@")[0])
        op_file = case.op_file if case else (row.get("op_file") or op)
        out.append(_entry(run, repo, op, op_file, statuses, row, indexes))
    return out


def write_from_gate(ab_dir: Path, cases,
                    gate: Dict[str, Dict[str, Dict[str, str]]]) -> Tuple[Path, Path, int]:
    """Rebuild both files from the gate's whole record; returns (md, draft, #ops).

    Always writes, even with no no_test left: a report listing ops that now pass would
    otherwise stay behind, and a resume that fixed them all has to say so."""
    entries = entries_from_gate(cases, gate)
    md_path, map_path = write_reports(ab_dir, entries)
    return md_path, map_path, len(entries)


def write_reports(ab_dir: Path, entries: List[Entry],
                  md_path: Optional[Path] = None,
                  map_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """Write the report and the commented draft; returns both paths."""
    md_path = md_path or ab_dir / MD_NAME
    map_path = map_path or ab_dir / MAP_NAME
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(_map_text(ab_dir, entries), encoding="utf-8")
    md_path.write_text(_md_text(ab_dir, entries), encoding="utf-8")
    return md_path, map_path


def _tag(hint: Hint) -> str:
    """The bracket shown for a candidate; `·不确定` marks a node that may not exist."""
    return hint.tag if hint.certain else f"{hint.tag}·不确定"


def _md_text(ab_dir: Path, entries: List[Entry]) -> str:
    with_hints = [e for e in entries if e.hints]
    without = [e for e in entries if not e.hints]
    counts = Counter(e.repo for e in entries)
    lines = [
        "# 精度门禁：未找到单测的算子",
        "",
        f"来源：`{ab_dir / MANIFEST}` 的 `accuracy.sides`（`status=no_test`），"
        f"生成于 {datetime.now():%Y-%m-%d %H:%M}。",
        "",
    ]
    if not entries:
        lines += [
            "当前记录里没有 `no_test` 算子：每个被门禁检查的算子都定位到了测试"
            "（或已被排除）。如果之前有一份候选清单，它已经被这份替换。",
            "",
        ]
        return "\n".join(lines)
    lines += [
        f"共 {len(entries)} 个算子（"
        + "、".join(f"{repo} {n}" for repo, n in sorted(counts.items()))
        + f"）：{len(with_hints)} 个找到候选，{len(without)} 个没有。",
        "",
        "候选只是**搜索**结果，不是结论：写错的 target 会让该算子判 fail，或者用一个"
        "不覆盖它的测试去门禁它。节点是按默认 pytest 配置**推断**的，不保证一定能收集"
        "（文件 import 失败、仓库改了 `python_classes` 等都会影响）。类里的用例带类路径"
        "（`TestA::test_x`）；类要是 pytest 认的测试容器——`Test*` 前缀（带 `__init__` 的会被跳过），"
        "或本文件 import 的 `unittest.TestCase` 子类（名字以 `TestCase` 结尾不算证据），"
        "`__test__ = False` 的不算。标了 `·不确定` 的要么写在 `if`/`try` 里（条件不成立时收集不到），"
        "要么基类解析不了（可能是别处 import 的 TestCase 子类），"
        "`ab_run.py` 的自动匹配不会选它。确认哪一条对之后，把它写成一行、存成 map 文件，再用 "
        "`--accuracy-map` 传入（每行 `<[repo:]op[@op_file]> <tests/...py[::test_func]>`）：",
        "",
        "```",
        "main:add_rms_norm tests/test_add_rms_norm.py::test_add_rms_norm_accuracy",
        "```",
        "",
        f"同一批候选的可编辑版本：`{MAP_NAME}`（说明行以 `###` 开头，候选行只带一个 `#`；"
        "`###` 里写着匹配档位与依据，去掉候选行的 `#` 即可用，每个算子只留一行）。",
        "",
    ]
    if with_hints:
        lines += ["## 有候选", ""]
        for e in with_hints:
            lines += [f"### `{e.repo}:{e.op}@{e.op_file}`", "",
                      f"- benchmark：`{e.bench or '(inventory 里没有这一行)'}`",
                      "- 门禁记录：" + "，".join(
                          f"{side} `{st}`" for side, st in e.sides.items() if st),
                      "",
                      "候选：", ""]
            for i, h in enumerate(e.hints, 1):
                lines += [f"{i}. `{h.target}`", f"   [{_tag(h)}] {h.why}"]
            lines.append("")
    if without:
        lines += ["## 没有候选", "",
                  "`tests/` 下没有文件名或函数名与这些算子相近的用例：可能确实没有测试，"
                  "也可能测试名与算子名毫无关系（`--accuracy-map` 就是为这种准备的）。",
                  ""]
        for e in without:
            lines.append(f"- `{e.repo}:{e.op}@{e.op_file}`"
                         f"（benchmark `{e.bench or '?'}`）")
        lines.append("")
    return "\n".join(lines)


def _map_text(ab_dir: Path, entries: List[Entry]) -> str:
    """The draft, with two comment levels: `###` prose, `#` a candidate line.

    Only a candidate line becomes a map entry by deleting its one `#`, so the prose
    (headers, tier notes, the per-op banner) can never be enabled by accident -- and a
    prose line one `#` short is still `##...`, still a comment to load_accuracy_map."""
    lines = [
        f"### ab_run.py --accuracy-gate 自动生成：no_test 算子的候选测试（{ab_dir.name}，"
        f"{datetime.now():%Y-%m-%d %H:%M}）",
        "### 说明行一律以 '###' 开头；候选行只带一个 '#'，去掉它这一行就生效（每个算子只留一行，"
        "同一个算子映射两次会在启动时被拒）。",
        "### 格式：<[repo:]op[@op_file]> <tests/...py[::test_func]>",
        "### 档位：[同名文件] 最可信；[文件名相近]/[文件名包含] 只是名字像；"
        "[函数名命中] 只看函数名，文件可能与算子无关；[benchmark 名命中] 最弱，"
        "只按 benchmark 文件名命中函数名；·不确定 = 写在 if/try 里，或类是否被 pytest 收集无法确认。",
        "",
    ]
    if not entries:
        lines += ["### 当前记录里没有 no_test 算子：这份清单是空的（旧内容已被替换）。", ""]
        return "\n".join(lines)
    for e in entries:
        key = f"{e.repo}:{e.op}@{e.op_file}"
        lines.append(f"### {key}  —  {e.bench or '?'}")
        if not e.hints:
            lines += ["###   没有候选：tests/ 下没有名字相近的文件或用例", ""]
            continue
        for h in e.hints[:MAX_MAP]:
            lines += [f"### [{_tag(h)}] {h.why}", f"# {key} {h.target}"]
        if len(e.hints) > MAX_MAP:
            lines.append(f"### ... 另有 {len(e.hints) - MAX_MAP} 个候选，见 "
                         f"{MD_NAME}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ab-dir", type=Path, required=True,
                   help="ab_run.py output dir (reads ab.json's accuracy.sides)")
    p.add_argument("--out-md", type=Path, default=None,
                   help=f"report path (default <ab-dir>/{MD_NAME})")
    p.add_argument("--out-map", type=Path, default=None,
                   help=f"`###` prose / one-`#`-per-candidate --accuracy-map draft "
                        f"(default <ab-dir>/{MAP_NAME})")
    args = p.parse_args()

    ab_dir = args.ab_dir.resolve()
    manifest = ab_dir / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"{LOG} no {manifest}: give an ab_run.py output dir")
    entries = build_entries(ab_dir)
    # Write even with nothing left to resolve: the old list must not survive a run that
    # fixed them all.
    md_path, map_path = write_reports(ab_dir, entries, args.out_md, args.out_map)
    hinted = sum(1 for e in entries if e.hints)
    print(f"{LOG} {len(entries)} ops without a test ({hinted} with candidates) -> "
          f"{md_path} and {map_path}", flush=True)


if __name__ == "__main__":
    main()
