"""Formula IR + equivalence engines for extracted NP4G programs.

The extractor (``extract_np4g_program.py``) turns a hard-forward NP4G circuit
into a *program*: per-tick channel predicates given as Boolean functions over

  * binary tasks: pair-state bits, transposed pair-state bits, EXISTS-join bits
    and raw relation bits, composed over ticks;
  * unary tasks: node-state bits, EXISTS/FORALL/saturating-count
    aggregates of state bits over input relations, global-EXISTS bits.

This module gives that program a first-order semantics and decides whether it
is EQUIVALENT to the intended target logic:

  * ``PairExpr`` / ``NodeExpr``: a tiny FO fragment (relation atoms between the
    two implicit arguments, equality, Boolean ops, ``Join`` = EXISTS over a
    middle object, EXISTS/FORALL/count-threshold neighbourhood quantifiers,
    global EXISTS).
  * ``eval_pair`` / ``eval_node``: exact evaluation on an explicit finite
    structure (used for cross-checking formula vs. model and for falsification
    at large n).
  * UCQ engine: formulas that normalise to unions of conjunctive queries with
    (possibly negated) unary atoms, equality and inequality atoms are decided
    EXACTLY via the classical canonical-database / homomorphism containment
    argument (extended to unary types by treating them as colours and to
    (in)equalities by enumerating variable identifications).
  * counting engine: unary formulas that normalise to Boolean combinations of
    ``count_{r}(x) in S`` atoms over a single relation are decided exactly on
    the count abstraction.

Everything outside those fragments is reported honestly as NOT PROVED.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np


# ============================================================================
# Formula IR
# ============================================================================
# Binary (pair) formulas denote predicates F(x, y) over a finite structure
# whose binary relations are numpy bool arrays rel[name][i, j] and whose unary
# predicates are numpy bool arrays un[name][i].
#
#   Rel(name, swapped)     -> rel[name](x,y) (or (y,x) if swapped)
#   Un(name, arg, neg)     -> un[name](x) / un[name](y), optionally negated
#   Eq(swapped)            -> x == y
#   Const(v), And, Or, Not
#   Join(left, right)      -> EXISTS k: left(x,k) AND right(k,y)
#
# Unary (node) formulas denote predicates G(x):
#
#   UnSelf(name, neg)          -> un[name](x)
#   NExists(rel, incoming, inner)  -> EXISTS j: rel(x,j) [or rel(j,x)] AND inner(j)
#   NForall(rel, incoming, inner)  -> FORALL j with rel edge: inner(j)
#   NCount(rel, incoming, inner, allowed)
#                              -> |{j: rel-edge AND inner(j)}| in allowed
#                                 (allowed is a set of ints; counts above CAP
#                                  are truncated by the extractor before here)
#   GExists(inner)             -> EXISTS z: inner(z)
#   Const, And, Or, Not


class Expr:
    """Base class; immutable, hashable by structural key."""
    _key: tuple

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expr) and self._key == other._key


def _fmt_args(items: Iterable[Any]) -> str:
    return ", ".join(str(i) for i in items)


class Const(Expr):
    def __init__(self, value: bool) -> None:
        self.value = bool(value)
        self._key = ("const", self.value)

    def __repr__(self) -> str:
        return "TRUE" if self.value else "FALSE"


TRUE = Const(True)
FALSE = Const(False)


class And(Expr):
    def __init__(self, *terms: Expr) -> None:
        self.terms = tuple(terms)
        self._key = ("and", tuple(t._key for t in self.terms))

    def __repr__(self) -> str:
        return "(" + " AND ".join(map(repr, self.terms)) + ")"


class Or(Expr):
    def __init__(self, *terms: Expr) -> None:
        self.terms = tuple(terms)
        self._key = ("or", tuple(t._key for t in self.terms))

    def __repr__(self) -> str:
        return "(" + " OR ".join(map(repr, self.terms)) + ")"


class Not(Expr):
    def __init__(self, term: Expr) -> None:
        self.term = term
        self._key = ("not", term._key)

    def __repr__(self) -> str:
        return f"NOT {self.term!r}"


class Rel(Expr):
    def __init__(self, name: str, swapped: bool = False) -> None:
        self.name = name
        self.swapped = bool(swapped)
        self._key = ("rel", name, self.swapped)

    def __repr__(self) -> str:
        return f"{self.name}(y,x)" if self.swapped else f"{self.name}(x,y)"


class Un(Expr):
    """Unary atom inside a PAIR formula: applies to x (arg=0) or y (arg=1)."""

    def __init__(self, name: str, arg: int) -> None:
        self.name = name
        self.arg = int(arg)
        self._key = ("un", name, self.arg)

    def __repr__(self) -> str:
        return f"{self.name}({'xy'[self.arg]})"


class Eq(Expr):
    def __init__(self) -> None:
        self._key = ("eq",)

    def __repr__(self) -> str:
        return "x==y"


class Join(Expr):
    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right
        self._key = ("join", left._key, right._key)

    def __repr__(self) -> str:
        return f"EXISTS k[{self.left!r}](x,k) AND [{self.right!r}](k,y)"


class UnSelf(Expr):
    def __init__(self, name: str) -> None:
        self.name = name
        self._key = ("unself", name)

    def __repr__(self) -> str:
        return f"{self.name}(x)"


class NodeAt(Expr):
    """Pair-level application of a NODE formula at x (arg=0) or y (arg=1)."""

    def __init__(self, inner: Expr, arg: int) -> None:
        self.inner = inner
        self.arg = int(arg)
        self._key = ("nodeat", inner._key, self.arg)

    def __repr__(self) -> str:
        return f"[{self.inner!r}]({'xy'[self.arg]})"


def _rel_key(rel) -> tuple:
    return ("relname", rel) if isinstance(rel, str) else ("relexpr", rel._key)


def _rel_repr(rel, a: str, b: str) -> str:
    return f"{rel}({a},{b})" if isinstance(rel, str) else f"[{rel!r}]({a},{b})"


class NExists(Expr):
    """EXISTS j: edge(x,j) AND inner(j). `rel` may be a relation NAME or a
    PAIR formula (used when substituting an upstream ADF predicate)."""

    def __init__(self, rel, incoming: bool, inner: Expr) -> None:
        self.rel = rel
        self.incoming = bool(incoming)
        self.inner = inner
        self._key = ("nexists", _rel_key(rel), self.incoming, inner._key)

    def __repr__(self) -> str:
        edge = _rel_repr(self.rel, "j", "x") if self.incoming else _rel_repr(self.rel, "x", "j")
        return f"EXISTS j: {edge} AND [{self.inner!r}](j)"


class NForall(Expr):
    def __init__(self, rel, incoming: bool, inner: Expr) -> None:
        self.rel = rel
        self.incoming = bool(incoming)
        self.inner = inner
        self._key = ("nforall", _rel_key(rel), self.incoming, inner._key)

    def __repr__(self) -> str:
        edge = _rel_repr(self.rel, "j", "x") if self.incoming else _rel_repr(self.rel, "x", "j")
        return f"FORALL j: {edge} -> [{self.inner!r}](j)"


class NCount(Expr):
    """count in `allowed`; counts are saturated at `cap` before membership."""

    def __init__(self, rel, incoming: bool, inner: Expr, allowed: frozenset[int], cap: int) -> None:
        self.rel = rel
        self.incoming = bool(incoming)
        self.inner = inner
        self.allowed = frozenset(int(a) for a in allowed)
        self.cap = int(cap)
        if self.cap < 0:
            raise ValueError("count saturation cap must be nonnegative")
        self._key = ("ncount", _rel_key(rel), self.incoming, inner._key, tuple(sorted(self.allowed)), self.cap)

    def __repr__(self) -> str:
        edge = _rel_repr(self.rel, "j", "x") if self.incoming else _rel_repr(self.rel, "x", "j")
        return f"|{{j: {edge} AND [{self.inner!r}](j)}}|@cap{self.cap} in {sorted(self.allowed)}"


class GExists(Expr):
    def __init__(self, inner: Expr) -> None:
        self.inner = inner
        self._key = ("gexists", inner._key)

    def __repr__(self) -> str:
        return f"EXISTS z: [{self.inner!r}](z)"


# ============================================================================
# Evaluation on explicit finite structures
# ============================================================================
# structure: {"rels": {name: bool[n,n]}, "uns": {name: bool[n]}}


def eval_pair(expr: Expr, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate a pair formula -> bool[n, n]."""
    n = next(iter(rels.values())).shape[0] if rels else next(iter(uns.values())).shape[0]
    if isinstance(expr, Const):
        return np.full((n, n), expr.value, dtype=bool)
    if isinstance(expr, Rel):
        v = rels[expr.name]
        return v.T.copy() if expr.swapped else v.copy()
    if isinstance(expr, Un):
        v = uns[expr.name]
        return np.broadcast_to(v[:, None] if expr.arg == 0 else v[None, :], (n, n)).copy()
    if isinstance(expr, Eq):
        return np.eye(n, dtype=bool)
    if isinstance(expr, Not):
        return ~eval_pair(expr.term, rels, uns)
    if isinstance(expr, And):
        out = np.full((n, n), True)
        for t in expr.terms:
            out &= eval_pair(t, rels, uns)
        return out
    if isinstance(expr, Or):
        out = np.full((n, n), False)
        for t in expr.terms:
            out |= eval_pair(t, rels, uns)
        return out
    if isinstance(expr, Join):
        left = eval_pair(expr.left, rels, uns)
        right = eval_pair(expr.right, rels, uns)
        # uint8 accumulation wraps at 256 witnesses and can therefore turn a
        # true existential join into false on the n=500 falsification cells.
        # float32 represents every possible witness count in this protocol
        # exactly while retaining NumPy's efficient matrix multiplication.
        return (left.astype(np.float32) @ right.astype(np.float32)) > 0.0
    if isinstance(expr, NodeAt):
        v = eval_node(expr.inner, rels, uns)
        return np.broadcast_to(v[:, None] if expr.arg == 0 else v[None, :],
                               (v.shape[0], v.shape[0])).copy()
    raise TypeError(f"not a pair expr: {expr!r}")


def _edge_matrix(rel, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> np.ndarray:
    return rels[rel] if isinstance(rel, str) else eval_pair(rel, rels, uns)


def eval_node(expr: Expr, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate a node formula -> bool[n]."""
    n = next(iter(rels.values())).shape[0] if rels else next(iter(uns.values())).shape[0]
    if isinstance(expr, Const):
        return np.full((n,), expr.value, dtype=bool)
    if isinstance(expr, UnSelf):
        return uns[expr.name].copy()
    if isinstance(expr, Not):
        return ~eval_node(expr.term, rels, uns)
    if isinstance(expr, And):
        out = np.full((n,), True)
        for t in expr.terms:
            out &= eval_node(t, rels, uns)
        return out
    if isinstance(expr, Or):
        out = np.full((n,), False)
        for t in expr.terms:
            out |= eval_node(t, rels, uns)
        return out
    if isinstance(expr, NExists):
        e = _edge_matrix(expr.rel, rels, uns)
        edge = e.T if expr.incoming else e
        inner = eval_node(expr.inner, rels, uns)
        return (edge & inner[None, :]).any(axis=1)
    if isinstance(expr, NForall):
        e = _edge_matrix(expr.rel, rels, uns)
        edge = e.T if expr.incoming else e
        inner = eval_node(expr.inner, rels, uns)
        return (~(edge & ~inner[None, :])).all(axis=1)
    if isinstance(expr, NCount):
        e = _edge_matrix(expr.rel, rels, uns)
        edge = e.T if expr.incoming else e
        inner = eval_node(expr.inner, rels, uns)
        counts = np.minimum((edge & inner[None, :]).sum(axis=1), expr.cap)
        allowed = np.zeros(expr.cap + 1, dtype=bool)
        for a in expr.allowed:
            if 0 <= a <= expr.cap:
                allowed[a] = True
        return allowed[counts]
    if isinstance(expr, GExists):
        inner = eval_node(expr.inner, rels, uns)
        return np.full((n,), bool(inner.any()), dtype=bool)
    raise TypeError(f"not a node expr: {expr!r}")


# ============================================================================
# Simplification
# ============================================================================


def simplify(expr: Expr) -> Expr:
    """Constant folding + flattening + dedup + basic absorption (recursive)."""
    if isinstance(expr, (Const, Rel, Un, Eq, UnSelf)):
        return expr
    if isinstance(expr, Not):
        t = simplify(expr.term)
        if isinstance(t, Const):
            return Const(not t.value)
        if isinstance(t, Not):
            return t.term
        return Not(t)
    if isinstance(expr, (And, Or)):
        is_and = isinstance(expr, And)
        flat: list[Expr] = []
        for t in expr.terms:
            s = simplify(t)
            if isinstance(s, Const):
                if s.value == is_and:
                    continue        # neutral element
                return Const(not is_and)  # absorbing element
            if isinstance(s, And if is_and else Or):
                flat.extend(s.terms)
            else:
                flat.append(s)
        seen: dict[tuple, Expr] = {}
        for t in flat:
            seen.setdefault(t._key, t)
        terms = list(seen.values())
        # complement check: p AND NOT p -> FALSE / p OR NOT p -> TRUE
        keys = set(seen.keys())
        for t in terms:
            if isinstance(t, Not) and t.term._key in keys:
                return Const(not is_and)
        # NCount over the same (rel, direction, inner) counts a single quantity,
        # so an AND of them is the count landing in the INTERSECTION of the
        # allowed sets (an OR is the union). An empty intersection is FALSE --
        # this is what lets the reachability filter prove that a combination of
        # per-channel count-clusters is unreachable.
        by_count: dict[tuple, list[NCount]] = {}
        rest: list[Expr] = []
        for t in terms:
            if isinstance(t, NCount):
                gk = ("ncount", _rel_key(t.rel), t.incoming, t.inner._key, t.cap)
                by_count.setdefault(gk, []).append(t)
            else:
                rest.append(t)
        merged: list[Expr] = []
        for group in by_count.values():
            allowed = set(group[0].allowed)
            for nc in group[1:]:
                allowed = (allowed & nc.allowed) if is_and else (allowed | nc.allowed)
            nc0 = group[0]
            if is_and and not allowed:
                return FALSE
            merged.append(NCount(nc0.rel, nc0.incoming, nc0.inner, frozenset(allowed), nc0.cap))
        terms = rest + merged
        if not terms:
            return Const(is_and)
        if len(terms) == 1:
            return terms[0]
        return (And if is_and else Or)(*terms)
    if isinstance(expr, Join):
        left = simplify(expr.left)
        right = simplify(expr.right)
        if isinstance(left, Const) and not left.value:
            return FALSE
        if isinstance(right, Const) and not right.value:
            return FALSE
        return Join(left, right)
    if isinstance(expr, NodeAt):
        inner = simplify(expr.inner)
        if isinstance(inner, Const):
            return inner
        return NodeAt(inner, expr.arg)
    if isinstance(expr, (NExists, NForall)):
        inner = simplify(expr.inner)
        rel = expr.rel if isinstance(expr.rel, str) else simplify(expr.rel)
        if isinstance(rel, Const) and not rel.value:
            return TRUE if isinstance(expr, NForall) else FALSE
        if isinstance(inner, Const):
            if isinstance(expr, NExists) and not inner.value:
                return FALSE
            if isinstance(expr, NForall) and inner.value:
                return TRUE
        cls = type(expr)
        return cls(rel, expr.incoming, inner)
    if isinstance(expr, NCount):
        inner = simplify(expr.inner)
        rel = expr.rel if isinstance(expr.rel, str) else simplify(expr.rel)
        if isinstance(rel, Const) and not rel.value:
            return Const(0 in expr.allowed)
        if isinstance(inner, Const) and not inner.value:
            return Const(0 in expr.allowed)
        if not expr.allowed:
            return FALSE
        if expr.allowed >= set(range(expr.cap + 1)):
            return TRUE
        return NCount(rel, expr.incoming, inner, expr.allowed, expr.cap)
    if isinstance(expr, GExists):
        inner = simplify(expr.inner)
        if isinstance(inner, Const):
            # over a non-empty domain EXISTS z TRUE = TRUE, EXISTS z FALSE = FALSE
            return inner
        return GExists(inner)
    raise TypeError(f"cannot simplify: {expr!r}")


# ============================================================================
# Truth-table -> formula (monotone-first minimization)
# ============================================================================


def table_to_expr(vars_: Sequence[Expr], table: np.ndarray) -> Expr:
    """Convert a truth table over `vars_` (bool[2^k], LSB = vars_[0]) to an Expr.

    Prefers a NEGATION-FREE cover when the function is monotone in every
    variable (typical for these predicates); otherwise falls back to a plain
    DNF with negative literals (the equivalence engines may then reject it,
    which is reported honestly downstream).
    """
    k = len(vars_)
    assert table.shape == (2 ** k,)
    if table.all():
        return TRUE
    if not table.any():
        return FALSE

    assign = np.zeros((2 ** k, k), dtype=bool)
    for i in range(k):
        assign[:, i] = (np.arange(2 ** k) >> i) & 1

    # drop irrelevant variables first
    relevant = []
    for i in range(k):
        hi = table[assign[:, i]]
        lo = table[~assign[:, i]]
        if not np.array_equal(hi, lo):
            relevant.append(i)
    if len(relevant) < k:
        idx = np.zeros(2 ** len(relevant), dtype=int)
        for j, i in enumerate(relevant):
            idx = idx + (((np.arange(2 ** len(relevant)) >> j) & 1) << i)
        return table_to_expr([vars_[i] for i in relevant], table[idx])

    monotone = all(
        not (table[~assign[:, i]] & ~table[assign[:, i]]).any()
        for i in range(k)
    )
    if monotone:
        # minimal true points = prime implicants with positive literals only
        minimal: list[np.ndarray] = []
        order = np.argsort(assign.sum(axis=1))
        for row in order:
            if not table[row]:
                continue
            a = assign[row]
            if any(((m & a) == m).all() for m in minimal):
                continue
            minimal.append(a)
        disj = []
        for m in minimal:
            lits = [vars_[i] for i in range(k) if m[i]]
            disj.append(And(*lits) if len(lits) != 1 else lits[0])
        return simplify(Or(*disj) if len(disj) != 1 else disj[0])

    # non-monotone fallback: prime-implicant cover (Quine-McCluskey combine +
    # greedy cover). Any cover of the true rows by prime implicants denotes the
    # same Boolean function; merging complementary minterm pairs removes the
    # negative literals a raw minterm DNF would carry into the UCQ engine
    # (e.g. (a AND NOT b) OR (NOT a AND b) OR (a AND b) -> a OR b, keeping a
    # provable head formula inside the decidable fragment).
    if k <= 12:
        minterms = [row for row in range(2 ** k) if table[row]]
        full = (1 << k) - 1
        # implicant (care_mask, values): row matches iff row & care_mask == values
        cur = {(full, m) for m in minterms}
        primes: set[tuple[int, int]] = set()
        while cur:
            nxt: set[tuple[int, int]] = set()
            combined: set[tuple[int, int]] = set()
            for mask, val in cur:
                for i in range(k):
                    bit = 1 << i
                    if not (mask & bit):
                        continue
                    other = (mask, val ^ bit)
                    if other in cur:
                        nxt.add((mask & ~bit, val & ~bit))
                        combined.add((mask, val))
                        combined.add(other)
            primes |= cur - combined
            cur = nxt
        remaining = set(minterms)
        chosen: list[tuple[int, int]] = []
        # deterministic greedy cover, fewest-literal implicants preferred
        prime_list = sorted(primes, key=lambda p: (bin(p[0]).count("1"), p))
        while remaining:
            best = max(prime_list,
                       key=lambda p: sum(1 for r in remaining if (r & p[0]) == p[1]))
            cov = {r for r in remaining if (r & best[0]) == best[1]}
            chosen.append(best)
            remaining -= cov
        disj = []
        for mask, val in chosen:
            lits = [vars_[i] if (val >> i) & 1 else Not(vars_[i])
                    for i in range(k) if (mask >> i) & 1]
            disj.append(And(*lits) if len(lits) != 1 else lits[0])
        return simplify(Or(*disj) if len(disj) != 1 else disj[0])

    # very wide cones: plain DNF over true rows (may leave the UCQ fragment,
    # which is then reported honestly downstream)
    disj = []
    for row in range(2 ** k):
        if not table[row]:
            continue
        lits = [
            vars_[i] if assign[row, i] else Not(vars_[i])
            for i in range(k)
        ]
        disj.append(And(*lits) if len(lits) != 1 else lits[0])
    return simplify(Or(*disj) if len(disj) != 1 else disj[0])


def boolean_minimize(expr: Expr, max_atoms: int = 16) -> Expr:
    """Exact Boolean re-expression of a formula's propositional skeleton.

    Quantified subformulas (Join / NExists / NForall / NCount / GExists /
    NodeAt) are treated as opaque atoms (after minimizing their own bodies
    recursively); the skeleton's truth table is then re-covered, preferring a
    NEGATION-FREE form when the function is monotone. This removes the
    redundant negative literals that value-predicate DNFs produce (e.g.
    (a AND NOT b) OR (NOT a AND b) OR (a AND b) -> a OR b), which would
    otherwise push a provable formula outside the UCQ fragment.
    """

    def minimize_inner(e: Expr) -> Expr:
        if isinstance(e, Join):
            return Join(boolean_minimize(e.left, max_atoms), boolean_minimize(e.right, max_atoms))
        if isinstance(e, NodeAt):
            return NodeAt(boolean_minimize(e.inner, max_atoms), e.arg)
        if isinstance(e, NExists):
            rel = e.rel if isinstance(e.rel, str) else boolean_minimize(e.rel, max_atoms)
            return NExists(rel, e.incoming, boolean_minimize(e.inner, max_atoms))
        if isinstance(e, NForall):
            rel = e.rel if isinstance(e.rel, str) else boolean_minimize(e.rel, max_atoms)
            return NForall(rel, e.incoming, boolean_minimize(e.inner, max_atoms))
        if isinstance(e, NCount):
            rel = e.rel if isinstance(e.rel, str) else boolean_minimize(e.rel, max_atoms)
            return NCount(rel, e.incoming, boolean_minimize(e.inner, max_atoms), e.allowed, e.cap)
        if isinstance(e, GExists):
            return GExists(boolean_minimize(e.inner, max_atoms))
        return e

    expr = simplify(expr)
    atom_list: list[Expr] = []
    orig_index: dict[tuple, int] = {}

    def collect(e: Expr) -> None:
        if isinstance(e, (And, Or)):
            for t in e.terms:
                collect(t)
        elif isinstance(e, Not):
            collect(e.term)
        elif isinstance(e, Const):
            pass
        elif e._key not in orig_index:
            orig_index[e._key] = len(atom_list)
            atom_list.append(minimize_inner(e))

    collect(expr)
    if not atom_list or len(atom_list) > max_atoms:
        return expr
    rows = 1 << len(atom_list)

    def eval_skel(e: Expr, bits: int) -> bool:
        if isinstance(e, Const):
            return e.value
        if isinstance(e, And):
            return all(eval_skel(t, bits) for t in e.terms)
        if isinstance(e, Or):
            return any(eval_skel(t, bits) for t in e.terms)
        if isinstance(e, Not):
            return not eval_skel(e.term, bits)
        return bool((bits >> orig_index[e._key]) & 1)

    table = np.array([eval_skel(expr, b) for b in range(rows)], dtype=bool)
    return simplify(table_to_expr(atom_list, table))


# ============================================================================
# UCQ engine (pair formulas)
# ============================================================================
# A CQ over variables {0..v-1} with distinguished (x=0, y=1):
#   atoms: ("rel", name, a, b) / ("un", name, a) / ("nun", name, a)  (negated unary)
#          ("eq", a, b) / ("neq", a, b)
# UCQ = list of CQs. Conversion can fail (negated binary atom, NForall, NCount)
# -> UCQFail raised, caller falls back.


class UCQFail(Exception):
    pass


@dataclass(frozen=True)
class CQ:
    n_vars: int
    atoms: frozenset[tuple]

    def __post_init__(self):
        if self.n_vars < 2:
            raise UCQFail("CQ reserves distinguished variables 0 and 1")
        for atom in self.atoms:
            if not atom or atom[0] not in ("rel", "un", "nun", "eq", "neq"):
                raise UCQFail(f"unsupported CQ atom: {atom}")
            tag = atom[0]
            if len(atom) != (4 if tag == "rel" else 3):
                raise UCQFail(f"malformed CQ atom: {atom}")
            indices = atom[2:] if tag in ("rel", "un", "nun") else atom[1:]
            if any(not isinstance(v, int) or not 0 <= v < self.n_vars for v in indices):
                raise UCQFail(f"invalid CQ variable: {atom}")

    def substitute(self, mapping: dict[int, int], n_vars: int) -> "CQ":
        out = set()
        for a in self.atoms:
            if a[0] == "rel":
                out.add(("rel", a[1], mapping[a[2]], mapping[a[3]]))
            elif a[0] in ("un", "nun"):
                out.add((a[0], a[1], mapping[a[2]]))
            elif a[0] in ("eq", "neq"):
                out.add((a[0], mapping[a[1]], mapping[a[2]]))
            else:
                raise AssertionError(a)
        return CQ(n_vars, frozenset(out))


def _cq_product(a: list[CQ], b: list[CQ]) -> list[CQ]:
    """Conjunction of two UCQs (variables of b are shifted past a's)."""
    out = []
    for qa in a:
        for qb in b:
            shift = qa.n_vars - 2  # shared distinguished vars 0,1
            mapping = {0: 0, 1: 1}
            for v in range(2, qb.n_vars):
                mapping[v] = v + shift
            qb2 = qb.substitute(mapping, qa.n_vars + qb.n_vars - 2)
            out.append(CQ(qb2.n_vars, qa.atoms | qb2.atoms))
    return out


_FAMILY_PARENT_RELS = ("father", "mother")


def family_class_axiom_reduce(expr: Expr) -> Expr:
    """Simplify a family-vocabulary pair formula using the generator's class
    integrity constraints.  Besides dropping negated binary atoms that a
    positive parent atom in the SAME conjunction makes redundant, orient a
    co-parent witness when the same conjunction excludes the opposite parent
    role at an endpoint.

    ``random_family_structure`` builds father/mother so that, on every model of
    the class:

      * father and mother are DISJOINT as relations
        (an edge (a,b) records b as a's father iff b is male, as mother iff b
        is female -- never both):        father(a,b) => NOT mother(a,b).
      * parent roles are GLOBALLY EXCLUSIVE for a person: if any child records
        b as a father, no child records b as a mother, and conversely.  This is
        the column-wise consequence of father(a,b) => male(b) and
        mother(a,b) => NOT male(b).
      * the parent relation is ANTISYMMETRIC / acyclic (a parent has a strictly
        earlier index):                  parent(a,b) => NOT parent(b,a),
        i.e. father/mother(a,b) => NOT father/mother(b,a).

    Both are SOUND: dropping such a negated atom never changes the formula's
    truth value on any model of the class, so proving the reduced formula
    equivalent to the target on the class proves the original equivalent too
    (docs/paper_guide.md 2.4(b): equivalence is judged on the generator's
    structure class). The reduction is only *used* to reach the decidable UCQ
    fragment; the extractor independently falsifies the reduced formula against
    generator ground truth. This finite check is not a soundness proof.

    Purely local: within one And node, a positive ``Rel(p)`` over argument pair
    (u,v) licenses dropping ``Not(Rel(q))`` when (disjoint) q is the other
    parent relation over the SAME (u,v), or (antisymmetric) q is any parent
    relation over the REVERSED (v,u). Argument pairs are encoded by the atom's
    ``swapped`` flag, which is exactly the local (0,1)/(1,0) orientation, so no
    global variable bookkeeping is needed.

    A second local pattern occurs in the staged husband program.  With
    ``parent = father OR mother``, a shared-child witness

        EXISTS k: parent(k,x) AND parent(k,y)

    together with ``NOT is_mother(x)`` and ``NOT is_father(y)`` fixes that same
    witness to ``father(k,x) AND mother(k,y)``.  Global role exclusivity then
    makes the two negated role tests redundant.  The symmetric orientations are
    handled as well.  Requiring the witness and exclusions in one ``And`` keeps
    this rewrite local and prevents an exclusion from being consumed across an
    ``Or`` branch.
    """

    def local_size(e: Expr) -> int:
        if isinstance(e, (Rel, Un, Eq, Const)):
            return 1
        if isinstance(e, Not):
            return 1 + local_size(e.term)
        if isinstance(e, (And, Or)):
            return 1 + sum(local_size(t) for t in e.terms)
        return 100  # quantified subformulas are not local Boolean cases

    def negate_local(e: Expr) -> Expr:
        if isinstance(e, Not):
            return e.term
        if isinstance(e, (And, Or)):
            constructor = Or if isinstance(e, And) else And
            return constructor(*(negate_local(t) for t in e.terms))
        return simplify(Not(e))

    constants_cache: dict[Expr, Expr] = {}

    def local_constant(e: Expr) -> Expr:
        # Refute e or its Boolean complement using the corrected containment
        # checker. For example, an acyclic parent relation forbids a two-edge
        # cycle, so its complement is TRUE. This is an exhaustive logical check,
        # not a rewrite justified by sampled family structures. Size is bounded
        # before DNF conversion; unsupported negations leave e unchanged.
        if not isinstance(e, (And, Or, Not)) or local_size(e) > 16:
            return e
        if e in constants_cache:
            return constants_cache[e]
        result = e
        for value, candidate in ((False, e), (True, negate_local(e))):
            try:
                queries = pair_to_ucq(candidate)
                if ucq_contained(queries, [], ["father", "mother"], ["male"],
                                 class_model=family_class_model)[0]:
                    result = Const(value)
                    break
            except UCQFail:
                pass
        constants_cache[e] = result
        return result

    def parent_union(e: Expr, *, swapped: bool) -> bool:
        e = simplify(e)
        if not isinstance(e, Or):
            return False
        expected = {
            Rel("father", swapped)._key,
            Rel("mother", swapped)._key,
        }
        return {term._key for term in e.terms} == expected

    def role_exclusion(e: Expr) -> tuple[int, str] | None:
        if not isinstance(e, Not) or not isinstance(e.term, NodeAt):
            return None
        node = e.term.inner
        if not isinstance(node, NExists):
            return None
        if (
            not isinstance(node.rel, str)
            or node.rel not in _FAMILY_PARENT_RELS
            or not node.incoming
            or simplify(node.inner) != TRUE
        ):
            return None
        return e.term.arg, node.rel

    def orient_parent_join(
        e: Join,
        exclusions: set[tuple[int, str]],
    ) -> tuple[Join, set[tuple[int, str]]]:
        left, right = e.left, e.right
        consumed: set[tuple[int, str]] = set()
        if parent_union(left, swapped=True):
            if (0, "mother") in exclusions:
                left = Rel("father", True)
                consumed.add((0, "mother"))
            elif (0, "father") in exclusions:
                left = Rel("mother", True)
                consumed.add((0, "father"))
        if parent_union(right, swapped=False):
            if (1, "mother") in exclusions:
                right = Rel("father")
                consumed.add((1, "mother"))
            elif (1, "father") in exclusions:
                right = Rel("mother")
                consumed.add((1, "father"))
        return Join(left, right), consumed

    def rewrite(e: Expr) -> Expr:
        e = local_constant(e)
        if isinstance(e, And):
            terms = [rewrite(t) for t in e.terms]
            exclusions = {
                item for term in terms if (item := role_exclusion(term)) is not None
            }
            consumed_exclusions: set[tuple[int, str]] = set()
            oriented: list[Expr] = []
            for term in terms:
                if isinstance(term, Join):
                    term, consumed = orient_parent_join(term, exclusions)
                    consumed_exclusions.update(consumed)
                oriented.append(term)
            terms = [
                term
                for term in oriented
                if role_exclusion(term) not in consumed_exclusions
            ]
            pos = [(t.name, t.swapped) for t in terms
                   if isinstance(t, Rel) and t.name in _FAMILY_PARENT_RELS]
            kept: list[Expr] = []
            for t in terms:
                if (isinstance(t, Not) and isinstance(t.term, Rel)
                        and t.term.name in _FAMILY_PARENT_RELS):
                    nn, nsw = t.term.name, t.term.swapped
                    drop = False
                    for pn, psw in pos:
                        if psw == nsw and pn != nn:
                            drop = True           # disjointness (same args)
                            break
                        if psw != nsw:
                            drop = True           # antisymmetry (reversed args)
                            break
                    if drop:
                        continue                  # redundant on class => TRUE
                kept.append(t)
            if not kept:
                return TRUE
            return simplify(And(*kept)) if len(kept) > 1 else kept[0]
        if isinstance(e, Or):
            return simplify(Or(*[rewrite(t) for t in e.terms]))
        if isinstance(e, Not):
            return Not(rewrite(e.term))
        if isinstance(e, Join):
            return Join(rewrite(e.left), rewrite(e.right))
        if isinstance(e, NodeAt):
            return NodeAt(rewrite(e.inner), e.arg)
        if isinstance(e, NExists):
            rel = e.rel if isinstance(e.rel, str) else rewrite(e.rel)
            return NExists(rel, e.incoming, rewrite(e.inner))
        if isinstance(e, NForall):
            rel = e.rel if isinstance(e.rel, str) else rewrite(e.rel)
            return NForall(rel, e.incoming, rewrite(e.inner))
        if isinstance(e, NCount):
            rel = e.rel if isinstance(e.rel, str) else rewrite(e.rel)
            return NCount(rel, e.incoming, rewrite(e.inner), e.allowed, e.cap)
        if isinstance(e, GExists):
            return GExists(rewrite(e.inner))
        return e

    return simplify(rewrite(simplify(expr)))


def pair_to_ucq(expr: Expr) -> list[CQ]:
    """Pair formula -> UCQ with distinguished vars x=0, y=1. Raises UCQFail."""
    expr = simplify(expr)
    if isinstance(expr, Const):
        return [CQ(2, frozenset())] if expr.value else []
    if isinstance(expr, Rel):
        a, b = (1, 0) if expr.swapped else (0, 1)
        return [CQ(2, frozenset({("rel", expr.name, a, b)}))]
    if isinstance(expr, Un):
        return [CQ(2, frozenset({("un", expr.name, expr.arg)}))]
    if isinstance(expr, Eq):
        return [CQ(2, frozenset({("eq", 0, 1)}))]
    if isinstance(expr, Not):
        t = expr.term
        if isinstance(t, Un):
            return [CQ(2, frozenset({("nun", t.name, t.arg)}))]
        if isinstance(t, Eq):
            return [CQ(2, frozenset({("neq", 0, 1)}))]
        raise UCQFail(f"negation outside fragment: {expr!r}")
    if isinstance(expr, Or):
        out: list[CQ] = []
        for t in expr.terms:
            out.extend(pair_to_ucq(t))
        return out
    if isinstance(expr, And):
        qs = [CQ(2, frozenset())]
        for t in expr.terms:
            qs = _cq_product(qs, pair_to_ucq(t))
        return qs
    if isinstance(expr, Join):
        left = pair_to_ucq(expr.left)
        right = pair_to_ucq(expr.right)
        out = []
        for ql in left:
            for qr in right:
                # fresh middle variable k; left is (x,k), right is (k,y)
                # left: keep x=0, remap its y(=1) -> k; right: remap its x(=0) -> k
                k = 2
                lmap = {0: 0, 1: k}
                for v in range(2, ql.n_vars):
                    lmap[v] = v + 1
                nl = ql.n_vars + 1
                ql2 = ql.substitute(lmap, nl)
                rmap = {0: k, 1: 1}
                for v in range(2, qr.n_vars):
                    rmap[v] = v + nl - 2
                qr2 = qr.substitute(rmap, nl + qr.n_vars - 2)
                out.append(CQ(qr2.n_vars, ql2.atoms | qr2.atoms))
        return out
    if isinstance(expr, NodeAt):
        qs = node_to_ucq(expr.inner)
        if expr.arg == 0:
            return qs
        out = []
        for q in qs:
            mapping = {0: 1, 1: 0}
            for v in range(2, q.n_vars):
                mapping[v] = v
            out.append(q.substitute(mapping, q.n_vars))
        return out
    raise UCQFail(f"outside UCQ fragment: {expr!r}")


def _pair_ucq_between(expr: Expr, a: int, b: int, n_vars: int) -> list[CQ]:
    """UCQ of a pair formula with its (x, y) mapped to existing vars (a, b)."""
    out = []
    for q in pair_to_ucq(expr):
        mapping = {0: a, 1: b}
        for v in range(2, q.n_vars):
            mapping[v] = n_vars + (v - 2)
        out.append(q.substitute(mapping, n_vars + q.n_vars - 2))
    return out


def node_to_ucq(expr: Expr) -> list[CQ]:
    """Node formula -> UCQ with distinguished var x=0 (var 1 is an unused dummy,
    so the pair containment machinery can be reused unchanged)."""
    expr = simplify(expr)
    if isinstance(expr, Const):
        return [CQ(2, frozenset())] if expr.value else []
    if isinstance(expr, UnSelf):
        return [CQ(2, frozenset({("un", expr.name, 0)}))]
    if isinstance(expr, Not):
        t = expr.term
        if isinstance(t, UnSelf):
            return [CQ(2, frozenset({("nun", t.name, 0)}))]
        raise UCQFail(f"negation outside fragment: {expr!r}")
    if isinstance(expr, Or):
        out: list[CQ] = []
        for t in expr.terms:
            out.extend(node_to_ucq(t))
        return out
    if isinstance(expr, And):
        qs = [CQ(2, frozenset())]
        for t in expr.terms:
            qs = _cq_product(qs, node_to_ucq(t))
        return qs
    if isinstance(expr, NExists):
        return _ncount_to_ucq(expr.rel, expr.incoming, expr.inner, k_witnesses=1)
    if isinstance(expr, NForall):
        raise UCQFail(f"FORALL outside UCQ fragment: {expr!r}")
    if isinstance(expr, NCount):
        # upward-closed allowed set (w.r.t. the saturating cap) = "count >= k"
        ks = sorted(v for v in expr.allowed if 0 <= v <= expr.cap)
        if ks and all(v in expr.allowed for v in range(ks[0], expr.cap + 1)):
            return _ncount_to_ucq(expr.rel, expr.incoming, expr.inner, k_witnesses=ks[0])
        raise UCQFail(f"non-monotone count outside UCQ fragment: {expr!r}")
    if isinstance(expr, GExists):
        inner = node_to_ucq(expr.inner)
        out = []
        for q in inner:
            # existentially close the old root: root -> fresh var, new root 0 is free
            z = q.n_vars
            mapping = {0: z, 1: 1}
            for v in range(2, q.n_vars):
                mapping[v] = v
            out.append(q.substitute(mapping, q.n_vars + 1))
        return out
    raise UCQFail(f"outside UCQ node fragment: {expr!r}")


def _ncount_to_ucq(rel, incoming: bool, inner: Expr, k_witnesses: int) -> list[CQ]:
    """|{j : edge & inner(j)}| >= k as a CQ with k pairwise-distinct witnesses.

    `rel` may be a relation name (edge atom) or a pair formula (its UCQ is
    inlined between the root and the witness).
    """
    if k_witnesses == 0:
        return [CQ(2, frozenset())]
    if k_witnesses > 4:
        raise UCQFail(f"count >= {k_witnesses} too large for witness expansion")
    if not isinstance(rel, str) and k_witnesses > 1:
        raise UCQFail("count >= 2 over a composite relation is outside the fragment")
    inner_qs = node_to_ucq(inner)

    def witness_cqs(base: CQ) -> list[tuple[CQ, int]]:
        """Attach one fresh witness j to `base`."""
        out = []
        for iq in inner_qs:
            j = base.n_vars
            n1 = base.n_vars + 1
            mapping = {0: j, 1: 1}
            for v in range(2, iq.n_vars):
                mapping[v] = n1 + (v - 2)
            iq2 = iq.substitute(mapping, n1 + iq.n_vars - 2)
            if isinstance(rel, str):
                edge_atom = ("rel", rel, j, 0) if incoming else ("rel", rel, 0, j)
                out.append((CQ(iq2.n_vars, base.atoms | iq2.atoms | {edge_atom}), j))
            else:
                a, b = (j, 0) if incoming else (0, j)
                for eq in _pair_ucq_between(rel, a, b, iq2.n_vars):
                    out.append((CQ(eq.n_vars, base.atoms | iq2.atoms | eq.atoms), j))
        return out

    stack: list[tuple[CQ, list[int]]] = [(CQ(2, frozenset()), [])]
    for _ in range(k_witnesses):
        new_stack: list[tuple[CQ, list[int]]] = []
        for base, js in stack:
            for q2, j in witness_cqs(base):
                new_stack.append((q2, js + [j]))
        stack = new_stack
    results: list[CQ] = []
    for q, js in stack:
        neq = {("neq", a, b) for i, a in enumerate(js) for b in js[i + 1:]}
        results.append(CQ(q.n_vars, q.atoms | neq))
    return results


def _canonical_structures(q: CQ, un_names: list[str], *, colour_names: list[str],
                          enumerate_quotients: bool):
    """Quotients/colourings of q's canonical database consistent with eq/neq atoms.

    Yield (rels, uns, x_elem, y_elem, size). The containment checker always
    enables all quotients and colours every declared unary predicate; selecting
    fewer cases is not sufficient in the presence of class constraints.
    """
    n = q.n_vars
    # forced merges from eq atoms
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for atom in q.atoms:
        if atom[0] == "eq":
            ra, rb = find(atom[1]), find(atom[2])
            if ra != rb:
                parent[ra] = rb
    roots = sorted({find(v) for v in range(n)})
    base_map = {v: roots.index(find(v)) for v in range(n)}
    m = len(roots)

    neq_pairs = {(base_map[a], base_map[b]) for t, a, b in
                 ((at[0], at[1], at[2]) for at in q.atoms if at[0] == "neq")}
    if any(a == b for a, b in neq_pairs):
        return []  # contradictory disjunct

    partitions = (
        _partitions(m) if enumerate_quotients
        else [[[e] for e in range(m)]]  # finest quotient only
    )
    for partition in partitions:
        pmap = {}
        for gi, group in enumerate(partition):
            for e in group:
                pmap[e] = gi
        if any(pmap[a] == pmap[b] for a, b in neq_pairs):
            continue
        size = len(partition)
        full = {v: pmap[base_map[v]] for v in range(n)}
        rels: dict[str, set[tuple[int, int]]] = {}
        pos_un: dict[str, set[int]] = {u: set() for u in un_names}
        neg_un: dict[str, set[int]] = {u: set() for u in un_names}
        ok = True
        for atom in q.atoms:
            if atom[0] == "rel":
                rels.setdefault(atom[1], set()).add((full[atom[2]], full[atom[3]]))
            elif atom[0] == "un":
                pos_un.setdefault(atom[1], set()).add(full[atom[2]])
            elif atom[0] == "nun":
                neg_un.setdefault(atom[1], set()).add(full[atom[2]])
        for u in set(pos_un) | set(neg_un):
            if pos_un.get(u, set()) & neg_un.get(u, set()):
                ok = False
                break
        if not ok:
            continue
        # Preserve explicit positive/negative atoms; enumerate every remaining
        # assignment for the requested predicates.
        free: list[tuple[str, int]] = []
        for u in colour_names:
            for e in range(size):
                if e not in pos_un.get(u, set()) and e not in neg_un.get(u, set()):
                    free.append((u, e))
        if len(free) > 16:
            raise UCQFail(f"too many free unary colourings ({len(free)})")
        for bits in range(2 ** len(free)):
            uns = {u: np.zeros(size, dtype=bool) for u in un_names}
            for u in un_names:
                for e in pos_un.get(u, set()):
                    uns[u][e] = True
            for i, (u, e) in enumerate(free):
                uns[u][e] = bool((bits >> i) & 1)
            rmats = {}
            for rname, edges in rels.items():
                mat = np.zeros((size, size), dtype=bool)
                for a, b in edges:
                    mat[a, b] = True
                rmats[rname] = mat
            yield rmats, uns, full[0], full[1], size


def _partitions(m: int):
    """Stream all set partitions of {0..m-1}, finest first."""
    if m == 0:
        yield []
        return
    for sub in _partitions(m - 1):
        e = m - 1
        yield [list(g) for g in sub] + [[e]]
        for i in range(len(sub)):
            yield [g + [e] if j == i else list(g) for j, g in enumerate(sub)]


def _eval_ucq_on(q_list: list[CQ], rels: dict[str, np.ndarray], uns: dict[str, np.ndarray],
                 rel_names: list[str], un_names: list[str], x: int, y: int,
                 size: int | None = None) -> bool:
    """Does the UCQ hold at (x, y) on the structure? Brute-force homomorphism search."""
    if size is None:
        if not rels and not uns:
            raise UCQFail("an empty vocabulary requires explicit domain size")
        size = next(iter(rels.values())).shape[0] if rels else next(iter(uns.values())).shape[0]
    full_rels = {r: rels.get(r, np.zeros((size, size), dtype=bool)) for r in rel_names}
    full_uns = {u: uns.get(u, np.zeros(size, dtype=bool)) for u in un_names}
    for q in q_list:
        if _cq_holds(q, full_rels, full_uns, x, y, size):
            return True
    return False


def _cq_holds(q: CQ, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray],
              x: int, y: int, size: int) -> bool:
    order = list(range(2, q.n_vars))
    assignment = {0: x, 1: y}

    def check_partial(v_new: int) -> bool:
        for atom in q.atoms:
            if atom[0] == "rel":
                _, name, a, b = atom
                if a in assignment and b in assignment and (a == v_new or b == v_new or v_new == -1):
                    if not rels[name][assignment[a], assignment[b]]:
                        return False
            elif atom[0] == "un":
                _, name, a = atom
                if a in assignment and (a == v_new or v_new == -1):
                    if not uns[name][assignment[a]]:
                        return False
            elif atom[0] == "nun":
                _, name, a = atom
                if a in assignment and (a == v_new or v_new == -1):
                    if uns[name][assignment[a]]:
                        return False
            elif atom[0] == "eq":
                _, a, b = atom
                if a in assignment and b in assignment and (a == v_new or b == v_new or v_new == -1):
                    if assignment[a] != assignment[b]:
                        return False
            elif atom[0] == "neq":
                _, a, b = atom
                if a in assignment and b in assignment and (a == v_new or b == v_new or v_new == -1):
                    if assignment[a] == assignment[b]:
                        return False
        return True

    if not check_partial(-1):
        return False

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        v = order[i]
        for e in range(size):
            assignment[v] = e
            if check_partial(v) and backtrack(i + 1):
                return True
            del assignment[v]
        return False

    return backtrack(0)


@dataclass(frozen=True)
class DenialConstraints:
    """Finite universal denials over the CQ literal grammar.

    Each clause is forbidden at every assignment. Restricting the domain and
    deleting binary edges preserves these constraints; unary values are fixed.
    """
    clauses: tuple[CQ, ...]

    def __call__(self, rels, uns):
        size = next(iter(rels.values())).shape[0] if rels else next(iter(uns.values())).shape[0]
        for clause in self.clauses:
            for x in range(size):
                for y in range(size):
                    if _cq_holds(clause, rels, uns, x, y, size):
                        return None
        return rels, uns


def family_class_model(rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]):
    """Pure filter: functional, irreflexive, disjoint, typed, acyclic parents.

    Unary assignments are TOTAL. In particular false is never an unspecified
    value to be overwritten. Acyclicity is preserved by domain restriction and
    edge deletion, so the canonical-substructure argument still applies.
    """
    father, mother = rels.get("father"), rels.get("mother")
    if father is None and mother is None:
        return rels, uns
    size = father.shape[0] if father is not None else mother.shape[0]
    empty = np.zeros((size, size), dtype=bool)
    father = father if father is not None else empty
    mother = mother if mother is not None else empty
    if any(np.diag(m).any() or (m.sum(axis=1) > 1).any() for m in (father, mother)):
        return None
    if (father & mother).any():
        return None
    if "male" not in uns:
        raise UCQFail("family typing requires a total male assignment")
    male = uns["male"]
    if (father.any(axis=0) & ~male).any() or (mother.any(axis=0) & male).any():
        return None
    reach = (father | mother).copy()
    for k in range(size):
        reach |= reach[:, k, None] & reach[None, k, :]
    if np.diag(reach).any():
        return None
    return rels, uns


def graph_class_model(rels, uns):
    """The experimental graph domain has no self edges."""
    adjacency = rels.get("adj")
    return None if adjacency is not None and np.diag(adjacency).any() else (rels, uns)


def _symbolic_contained(source: CQ, targets: list[CQ], family: bool = False) -> bool:
    """Sufficient containment by a literal-preserving homomorphism.

    Unlike testing a single concrete database, negative unary and inequality
    obligations must be entailed by SOURCE literals. A failed search falls back
    to exhaustive quotient/type enumeration. Family closure uses only equality
    from functionality and the declared parent typing/irreflexivity constraints.
    """
    parent = list(range(source.n_vars))
    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v
    def merge(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[b] = a
        return True
    for atom in source.atoms:
        if atom[0] == "eq":
            merge(atom[1], atom[2])
    if family:
        changed = True
        while changed:
            changed = False
            destinations = {}
            for atom in source.atoms:
                if atom[0] == "rel" and atom[1] in _FAMILY_PARENT_RELS:
                    key = atom[1], find(atom[2])
                    if key in destinations:
                        changed |= merge(destinations[key], atom[3])
                    else:
                        destinations[key] = atom[3]
    mapped = source.substitute({v: find(v) for v in range(source.n_vars)}, source.n_vars)
    atoms = set(mapped.atoms)
    if family:
        for atom in list(atoms):
            if atom[0] == "rel" and atom[1] in _FAMILY_PARENT_RELS:
                _, name, a, b = atom
                atoms.add(("neq", a, b))
                atoms.add(("un" if name == "father" else "nun", "male", b))
    for atom in list(atoms):
        if atom[0] == "neq":
            if atom[1] == atom[2]:
                return True  # source is unsatisfiable
            atoms.add(("neq", atom[2], atom[1]))
        if atom[0] == "un" and ("nun", atom[1], atom[2]) in atoms:
            return True
    domain = sorted({find(v) for v in range(source.n_vars)})
    visits = 0
    for target in targets:
        assignment = {0: find(0), 1: find(1)}
        def consistent():
            for atom in target.atoms:
                tag = atom[0]
                indices = atom[2:] if tag in ("rel", "un", "nun") else atom[1:]
                if not all(v in assignment for v in indices):
                    continue
                values = tuple(assignment[v] for v in indices)
                if tag == "eq":
                    if values[0] != values[1]:
                        return False
                else:
                    translated = ((tag, atom[1]) + values if tag in ("rel", "un", "nun")
                                  else (tag,) + values)
                    if translated not in atoms:
                        return False
            return True
        def search(v):
            nonlocal visits
            visits += 1
            if visits > 100000 or not consistent():
                return False
            if v == target.n_vars:
                return True
            for value in domain:
                assignment[v] = value
                if search(v + 1):
                    return True
            assignment.pop(v, None)
            return False
        if search(2):
            return True
    return False


def ucq_contained(q1: list[CQ], q2: list[CQ], rel_names: list[str], un_names: list[str],
                  class_model=None) -> tuple[bool, Any]:
    """Containment via all source quotients and complete unary assignments.

    Sound and complete if exhaustive enumeration finishes. Each countermodel
    satisfies its source and the declared constraints. The finite bound follows
    from retaining a source witness's domain, positive binary edges and exact
    unary values. Resource limits raise UCQFail, never acceptance.
    """
    if (class_model is not None and class_model not in (family_class_model, graph_class_model)
            and not isinstance(class_model, DenialConstraints)):
        raise UCQFail("unsupported constraint filter")
    rel_names, un_names = sorted(set(rel_names)), sorted(set(un_names))
    if class_model is family_class_model and "male" not in un_names:
        un_names.append("male")
    clauses = list(class_model.clauses) if isinstance(class_model, DenialConstraints) else []
    for q in q1 + q2 + clauses:
        for a in q.atoms:
            if a[0] == "rel" and a[1] not in rel_names:
                raise UCQFail(f"undeclared relation {a[1]}")
            if a[0] in ("un", "nun") and a[1] not in un_names:
                raise UCQFail(f"undeclared unary predicate {a[1]}")
    tested = 0
    for q in q1:
        if _symbolic_contained(q, q2, family=class_model is family_class_model):
            continue
        if q.n_vars > 16:
            raise UCQFail("quotient enumeration variable limit")
        for rels, uns, x, y, size in _canonical_structures(
            q, un_names, colour_names=un_names, enumerate_quotients=True
        ):
            tested += 1
            if tested > 1000000:
                raise UCQFail("canonical structure enumeration limit")
            rels = {r: rels.get(r, np.zeros((size, size), dtype=bool)) for r in rel_names}
            if class_model is not None:
                # Generic denials need an explicit domain even for an empty
                # vocabulary; q reserves free variables so size is positive.
                if not rels and not uns and isinstance(class_model, DenialConstraints):
                    if any(_cq_holds(c, {}, {}, a, b, size) for c in clauses
                           for a in range(size) for b in range(size)):
                        continue
                elif class_model(rels, uns) is None:
                    continue
            if not _eval_ucq_on([q], rels, uns, rel_names, un_names, x, y, size):
                raise UCQFail("canonical candidate does not satisfy source")
            if not _eval_ucq_on(q2, rels, uns, rel_names, un_names, x, y, size):
                return False, {"witness_rels": {k: v.tolist() for k, v in rels.items()},
                               "witness_uns": {k: v.tolist() for k, v in uns.items()},
                               "x": x, "y": y, "size": size}
    return True, None


def minimize_ucq(qs: list[CQ], rel_names: list[str], un_names: list[str]) -> list[CQ]:
    """Drop disjuncts contained in another single disjunct (cheap, sound)."""
    qs = list(dict.fromkeys(qs))
    kept: list[CQ] = []
    for i, q in enumerate(qs):
        others = qs[:i] + qs[i + 1:]
        dropped = False
        for o in others:
            if o in kept or o in qs[i + 1:]:
                ok, _ = ucq_contained([q], [o], rel_names, un_names)
                if ok:
                    dropped = True
                    break
        if not dropped:
            kept.append(q)
    return kept


def ucq_equivalent(q1: list[CQ], q2: list[CQ], rel_names: list[str], un_names: list[str],
                   class_model=None) -> tuple[bool, Any]:
    ok12, w12 = ucq_contained(q1, q2, rel_names, un_names, class_model=class_model)
    if not ok12:
        return False, {"direction": "extracted -> target fails", **w12}
    ok21, w21 = ucq_contained(q2, q1, rel_names, un_names, class_model=class_model)
    if not ok21:
        return False, {"direction": "target -> extracted fails", **w21}
    return True, None


# ============================================================================
# Counting engine (node formulas over a single relation's out/in-degree)
# ============================================================================


def count_saturation_bound(expr: Expr) -> int | None:
    """Largest atomic saturation point; None means unsupported grammar."""
    if isinstance(expr, Const):
        return 0
    if isinstance(expr, Not):
        return count_saturation_bound(expr.term)
    if isinstance(expr, (And, Or)):
        bounds = [count_saturation_bound(t) for t in expr.terms]
        return None if any(b is None for b in bounds) else max(bounds, default=0)
    if isinstance(expr, (NCount, NExists, NForall)):
        if not isinstance(expr.rel, str) or simplify(expr.inner) != TRUE:
            return None
        if isinstance(expr, NCount):
            return expr.cap if expr.cap >= 0 else None
        return 1 if isinstance(expr, NExists) else 0
    return None


def node_to_count_profile(expr: Expr, cap: int) -> tuple[str, bool, np.ndarray] | None:
    """If `expr` is a Boolean combination of NCount/NExists/NForall atoms over ONE
    (rel, direction) with inner == TRUE, return (rel, incoming, allowed[cap+1])
    where allowed[c] = truth value at saturated count c. Else None.
    """
    bound = count_saturation_bound(expr)
    if bound is None or cap < bound or cap < 0 or cap > 1000000:
        return None
    atoms: list[tuple[str, bool]] = []

    def scan(e: Expr) -> bool:
        if isinstance(e, Const):
            return True
        if isinstance(e, (And, Or)):
            return all(scan(t) for t in e.terms)
        if isinstance(e, Not):
            return scan(e.term)
        if isinstance(e, (NExists, NForall, NCount)):
            if simplify(e.inner) != TRUE:
                return False
            atoms.append((e.rel, e.incoming))
            return True
        return False

    if not scan(expr) or not atoms:
        return None
    keys = set(atoms)
    if len(keys) != 1:
        return None
    rel, incoming = next(iter(keys))

    def value_at(e: Expr, c: int) -> bool:
        if isinstance(e, Const):
            return e.value
        if isinstance(e, And):
            return all(value_at(t, c) for t in e.terms)
        if isinstance(e, Or):
            return any(value_at(t, c) for t in e.terms)
        if isinstance(e, Not):
            return not value_at(e.term, c)
        if isinstance(e, NExists):
            return c > 0
        if isinstance(e, NForall):
            return True  # inner TRUE -> vacuously/all true
        if isinstance(e, NCount):
            return min(c, e.cap) in e.allowed
        raise AssertionError(e)

    allowed = np.array([value_at(expr, c) for c in range(cap + 1)], dtype=bool)
    return rel, incoming, allowed


# ============================================================================
# Random structure generators (falsification / sanity checking)
# ============================================================================


def random_family_structure(rng: np.random.Generator, n: int, parent_prob: float = 0.95) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    male = rng.random(n) < 0.5
    father = np.zeros((n, n), dtype=bool)
    mother = np.zeros((n, n), dtype=bool)
    for child in range(1, n):
        older_male = np.flatnonzero(male[:child])
        older_female = np.flatnonzero(~male[:child])
        if older_male.size > 0 and rng.random() < parent_prob:
            father[child, int(rng.choice(older_male))] = True
        if older_female.size > 0 and rng.random() < parent_prob:
            mother[child, int(rng.choice(older_female))] = True
    parent = father | mother
    son = parent.T & male[None, :]
    daughter = parent.T & ~male[None, :]
    rels = {
        "father": father, "mother": mother,
        "son": son, "daughter": daughter,
    }
    uns = {"male": male}
    return rels, uns


def random_graph_structure(rng: np.random.Generator, n: int, p_min: float = 0.0, p_max: float = 0.3,
                           red_prob: float = 0.25) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    p = p_min + rng.random() * (p_max - p_min)
    adj = rng.random((n, n)) < p
    np.fill_diagonal(adj, False)
    red = rng.random(n) < red_prob
    rels = {"adj": adj, "eye": np.eye(n, dtype=bool)}
    uns = {"red": red}
    return rels, uns
