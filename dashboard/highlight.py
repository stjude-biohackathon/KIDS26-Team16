"""Marks verified quotes inside the note text, HTML-escaping everything else."""
from __future__ import annotations

import html
from typing import Any


def find_quote_spans(note_text: str, quotes: list[str]) -> list[tuple[int, int, str]]:
    """Finds non-overlapping match intervals for quotes in note_text, prioritizing longer quotes."""
    if not note_text:
        return []

    clean_quotes = sorted(
        {q.strip() for q in quotes if isinstance(q, str) and q.strip()},
        key=len,
        reverse=True,
    )
    lower_text = note_text.lower()
    occupied = [False] * len(note_text)
    spans: list[tuple[int, int, str]] = []

    for q in clean_quotes:
        q_lower = q.lower()
        q_len = len(q)
        start = 0
        while True:
            idx = lower_text.find(q_lower, start)
            if idx == -1:
                break
            end = idx + q_len
            if not any(occupied[idx:end]):
                spans.append((idx, end, q))
                for i in range(idx, end):
                    occupied[i] = True
            start = idx + 1

    spans.sort(key=lambda x: x[0])
    return spans


def highlight_note_quotes(note_text: str, findings: list[dict[str, Any]]) -> str:
    """Wraps matched verified quote substrings in <mark> tags with tooltip metadata."""
    if not note_text:
        return "<p class='text-muted'><em>No clinical note text available.</em></p>"

    quotes = [f.get("quote") for f in findings if isinstance(f, dict) and f.get("quote")]
    spans = find_quote_spans(note_text, quotes)

    if not spans:
        return (
            f"<div class='note-text-body' style='white-space: pre-wrap; font-family: ui-monospace, "
            f"\"SF Mono\", Menlo, monospace; font-size: 0.84rem; line-height: 1.65; "
            f"padding: 18px; border-radius: 6px; max-height: 480px; overflow-y: auto;'>"
            f"{html.escape(note_text)}</div>"
        )

    chunks: list[str] = []
    last_idx = 0
    for start, end, quote in spans:
        if start > last_idx:
            chunks.append(html.escape(note_text[last_idx:start]))
        matched_text = html.escape(note_text[start:end])
        safe_title = html.escape(f"Verified Quote: {quote}")
        chunks.append(
            f'<mark class="quote-highlight" title="{safe_title}">{matched_text}</mark>'
        )
        last_idx = end

    if last_idx < len(note_text):
        chunks.append(html.escape(note_text[last_idx:]))

    highlighted_body = "".join(chunks)
    return (
        f"<div class='note-text-body' style='white-space: pre-wrap; font-family: ui-monospace, "
        f"\"SF Mono\", Menlo, monospace; font-size: 0.84rem; line-height: 1.65; "
        f"padding: 18px; border-radius: 6px; max-height: 480px; overflow-y: auto;'>"
        f"{highlighted_body}</div>"
    )
