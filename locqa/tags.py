"""
Protected content inside segments.

XLIFF inline elements (<g>, <x/>, <ph>, <pc>, ...) are replaced by tokens such as ⟦1⟧ ... ⟦/1⟧
before a segment goes to the LLM, and put back verbatim when the file is written. Text
placeholders ({count}, {{name}}, %s, %1$s, ${var}) and HTML tags inside JSON strings stay in the
text as they are. Both kinds must appear in the translation exactly as in the source.
"""

from __future__ import annotations

import re
from collections import Counter
from xml.sax.saxutils import escape

TOKEN_RE = re.compile(r"⟦/?\d+⟧")
PLACEHOLDER_RE = re.compile(
    r"\{\{\s*[\w.]+\s*\}\}"          # {{name}}
    r"|\{[\w.]+(?:,[^{}]*)?\}"       # {count}, {n, plural, ...}
    r"|%(?:\d+\$)?[sdif@]"           # %s, %d, %1$s
    r"|\$\{\w+\}"                    # ${var}
    r"|</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>"  # <b>, </b>, <br/>
)


def visible(text: str) -> str:
    """The text a reader sees: tag tokens removed."""
    return TOKEN_RE.sub("", text)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(TOKEN_RE.sub(" ", text))


def protected(text: str) -> Counter:
    """Every tag token and placeholder in the text, with counts."""
    return Counter(tokens(text)) + Counter(placeholders(text))


def protected_diff(source: str, target: str) -> tuple[list[str], list[str]]:
    """(missing in target, unexpected in target)."""
    src, tgt = protected(source), protected(target)
    return sorted((src - tgt).elements()), sorted((tgt - src).elements())


def tag_order_problem(text: str) -> str | None:
    """Check that every ⟦/n⟧ closes an open ⟦n⟧ and paired tags are properly nested."""
    stack: list[str] = []
    closing_numbers = {t[2:-1] for t in tokens(text) if t.startswith("⟦/")}
    for tok in tokens(text):
        if tok.startswith("⟦/"):
            n = tok[2:-1]
            if not stack or stack[-1] != n:
                return f"tag {tok} closes before or outside its opening tag ⟦{n}⟧"
            stack.pop()
        elif tok[1:-1] in closing_numbers:
            stack.append(tok[1:-1])
    if stack:
        return f"tag ⟦{stack[-1]}⟧ is never closed"
    return None


def unmask_to_xml(text: str, tags: dict[str, str]) -> str:
    """Escape the text and put the original inline XML back in place of the tokens."""
    out: list[str] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        out.append(escape(text[pos:m.start()]))
        tok = m.group()
        if tok not in tags:
            raise ValueError(f"unknown tag token {tok}")
        out.append(tags[tok])
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)
