"""Predicate DSL: tokenize -> AST -> Kleene evaluation.

Decision-table rows are written as strings so they stay readable next to the
rubric text they encode. They are parsed at import time, so a malformed
predicate fails loudly at build time instead of silently at grading time, and
every identifier is checked against the feature registry.

Grammar
    expr    := or_expr
    or_expr := and_expr ("or" and_expr)*
    and_expr:= not_expr ("and" not_expr)*
    not_expr:= "not" not_expr | atom
    atom    := "(" expr ")" | comparison | NAME
    comparison := operand OP operand (OP operand)?      # chained: 2000 <= x <= 4999
                | NAME "in" "(" operand ("," operand)* ")"
    operand := NUMBER | NAME
    OP      := "<" | "<=" | ">" | ">=" | "==" | "!="

A bare NAME is a truth test on a boolean feature. In a comparison, a NAME on
the right-hand side is an enum *value* of the feature on the left.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- three-valued

class _Unknown:
    """Kleene UNKNOWN: the note does not say. Never conflate with False."""
    _inst = None
    def __new__(cls):
        if cls._inst is None: cls._inst = super().__new__(cls)
        return cls._inst
    def __bool__(self): raise TypeError("UNKNOWN is not a truth value; use kleene_* helpers")
    def __repr__(self): return "UNKNOWN"

UNKNOWN = _Unknown()

def k_not(a):
    return UNKNOWN if a is UNKNOWN else (not a)

def k_and(a, b):
    if a is False or b is False: return False        # short-circuits through UNKNOWN
    if a is UNKNOWN or b is UNKNOWN: return UNKNOWN
    return True

def k_or(a, b):
    if a is True or b is True: return True
    if a is UNKNOWN or b is UNKNOWN: return UNKNOWN
    return False

# ---------------------------------------------------------------------- tokens

TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>-?\d+(?:\.\d+)?)
  | (?P<op><=|>=|==|!=|<|>)
  | (?P<lp>\()
  | (?P<rp>\))
  | (?P<comma>,)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
""", re.X)

KEYWORDS = {"and", "or", "not", "in"}

# `20 <= mpap` is read as `mpap >= 20`
FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}

@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    pos: int

def tokenize(src: str) -> list[Tok]:
    out, i = [], 0
    while i < len(src):
        m = TOKEN.match(src, i)
        if not m:
            raise SyntaxError(f"unexpected character {src[i]!r} at {i} in {src!r}")
        i = m.end()
        kind = m.lastgroup
        if kind == "ws": continue
        text = m.group()
        if kind == "name" and text in KEYWORDS:
            kind = text
        out.append(Tok(kind, text, m.start()))
    return out

# ------------------------------------------------------------------------- AST

class Node:
    def evaluate(self, env: dict[str, Any], ctx) -> Any: raise NotImplementedError
    def names(self) -> set[str]: raise NotImplementedError

@dataclass(frozen=True)
class Truth(Node):
    """Bare feature name: true when the boolean feature is true."""
    name: str
    def evaluate(self, env, ctx):
        v = env.get(self.name, UNKNOWN)
        if v is UNKNOWN or v is None: return UNKNOWN
        return bool(v)
    def names(self): return {self.name}

@dataclass(frozen=True)
class Cmp(Node):
    """`feature OP operand`, where operand is a number or an enum value."""
    name: str
    op: str
    operand: Any            # float, or str (enum value)
    def evaluate(self, env, ctx):
        v = env.get(self.name, UNKNOWN)
        if v is UNKNOWN or v is None: return UNKNOWN
        return ctx.compare(self.name, v, self.op, self.operand)
    def names(self): return {self.name}

@dataclass(frozen=True)
class Between(Node):
    """`lo OP feature OP hi` - the chained form used for every rubric band."""
    lo: float
    lo_op: str
    name: str
    hi_op: str
    hi: float
    def evaluate(self, env, ctx):
        v = env.get(self.name, UNKNOWN)
        if v is UNKNOWN or v is None: return UNKNOWN
        a = ctx.compare(self.name, self.lo, self.lo_op, v, left_is_value=True)
        b = ctx.compare(self.name, v, self.hi_op, self.hi)
        return k_and(a, b)
    def names(self): return {self.name}

@dataclass(frozen=True)
class In(Node):
    name: str
    options: tuple
    negated: bool = False
    def evaluate(self, env, ctx):
        v = env.get(self.name, UNKNOWN)
        if v is UNKNOWN or v is None: return UNKNOWN
        hit = any(ctx.compare(self.name, v, "==", o) is True for o in self.options)
        return (not hit) if self.negated else hit
    def names(self): return {self.name}

@dataclass(frozen=True)
class Not(Node):
    child: Node
    def evaluate(self, env, ctx): return k_not(self.child.evaluate(env, ctx))
    def names(self): return self.child.names()

@dataclass(frozen=True)
class And(Node):
    children: tuple
    def evaluate(self, env, ctx):
        r = True
        for c in self.children: r = k_and(r, c.evaluate(env, ctx))
        return r
    def names(self): return set().union(*(c.names() for c in self.children))

@dataclass(frozen=True)
class Or(Node):
    children: tuple
    def evaluate(self, env, ctx):
        r = False
        for c in self.children: r = k_or(r, c.evaluate(env, ctx))
        return r
    def names(self): return set().union(*(c.names() for c in self.children))

# ---------------------------------------------------------------------- parser

class Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def eat(self, kind=None):
        t = self.peek()
        if t is None:
            raise SyntaxError(f"unexpected end of predicate in {self.src!r}")
        if kind and t.kind != kind:
            raise SyntaxError(f"expected {kind}, got {t.kind} {t.text!r} at {t.pos} in {self.src!r}")
        self.i += 1
        return t

    def parse(self) -> Node:
        node = self.or_expr()
        if self.peek() is not None:
            t = self.peek()
            raise SyntaxError(f"trailing {t.text!r} at {t.pos} in {self.src!r}")
        return node

    def or_expr(self):
        parts = [self.and_expr()]
        while self.peek() and self.peek().kind == "or":
            self.eat("or"); parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def and_expr(self):
        parts = [self.not_expr()]
        while self.peek() and self.peek().kind == "and":
            self.eat("and"); parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def not_expr(self):
        if self.peek() and self.peek().kind == "not":
            self.eat("not"); return Not(self.not_expr())
        return self.atom()

    def _operand(self):
        t = self.eat()
        if t.kind == "num": return float(t.text)
        if t.kind == "name": return t.text
        raise SyntaxError(f"expected value, got {t.text!r} at {t.pos} in {self.src!r}")

    def atom(self):
        t = self.peek()
        if t.kind == "lp":
            self.eat("lp"); node = self.or_expr(); self.eat("rp"); return node

        # value-first comparison: NUM OP NAME, optionally chained into a band
        # `2000 <= ferritin <= 4999` (Between) and `20 <= mpap` (reversed Cmp)
        if t.kind == "num":
            lo = float(self.eat("num").text)
            lo_op = self.eat("op").text
            name = self.eat("name").text
            nxt = self.peek()
            if nxt is not None and nxt.kind == "op":
                hi_op = self.eat("op").text
                hi = float(self.eat("num").text)
                return Between(lo, lo_op, name, hi_op, hi)
            return Cmp(name, FLIP[lo_op], lo)

        name = self.eat("name").text
        nxt = self.peek()
        if nxt is None or nxt.kind in {"and", "or", "rp"}:
            return Truth(name)
        if nxt.kind == "op":
            op = self.eat("op").text
            return Cmp(name, op, self._operand())
        if nxt.kind == "in":
            self.eat("in"); self.eat("lp")
            opts = [self._operand()]
            while self.peek() and self.peek().kind == "comma":
                self.eat("comma"); opts.append(self._operand())
            self.eat("rp")
            return In(name, tuple(opts))
        raise SyntaxError(f"unexpected {nxt.text!r} after {name!r} in {self.src!r}")

def parse(src: str) -> Node:
    return Parser(src).parse()
