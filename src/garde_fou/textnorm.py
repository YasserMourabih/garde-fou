"""Text normalisation.

Normalisation is the cheapest prompt-injection defence that exists: most
obfuscation attacks (zero-width characters, homoglyphs, leetspeak, letter
spacing) only survive because the detector reads the raw bytes. We therefore
expose it as an explicit, ablatable component rather than burying it inside a
single detector.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that are invisible but shift tokenisation.
INVISIBLE = dict.fromkeys(
    map(
        ord,
        "​‌‍‎‏⁠﻿­᠎⁡⁢⁣",
    )
)

# Latin look-alikes from other scripts, used to smuggle keywords past filters.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ν": "v", "ο": "o", "ρ": "p",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "τ": "t", "υ": "u", "Ι": "I",
    "Ο": "O", "Α": "A", "Ε": "E", "Ѕ": "S", "А": "A", "В": "B", "Е": "E",
    "К": "K", "М": "M", "Н": "H", "Р": "P", "С": "C", "Т": "T", "Х": "X",
}

LEET = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})

_SPACED_LETTERS = re.compile(r"\b(?:[^\W\d_]\s+){2,}[^\W\d_]\b", re.UNICODE)
_PUNCT_LETTERS = re.compile(r"\b(?:[^\W\d_][.\-_*|]){2,}[^\W\d_]\b", re.UNICODE)
_WS = re.compile(r"\s+")


def _collapse_spaced(text: str) -> str:
    """`i g n o r e` -> `ignore`, `i.g.n.o.r.e` -> `ignore`."""
    text = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)
    return _PUNCT_LETTERS.sub(lambda m: re.sub(r"[.\-_*|]", "", m.group(0)), text)


def normalize(text: str, *, aggressive: bool = True) -> str:
    """Canonicalise `text` so that obfuscated keywords become searchable.

    `aggressive=False` keeps only the lossless steps (NFKC, invisible-character
    removal, whitespace collapsing) and is what we use before *training*
    statistical models; `aggressive=True` additionally undoes homoglyphs,
    leetspeak and letter spacing and is what the rule engine reads.
    """
    text = unicodedata.normalize("NFKC", text).translate(INVISIBLE)
    if aggressive:
        text = "".join(HOMOGLYPHS.get(ch, ch) for ch in text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = _collapse_spaced(text)
        text = text.lower().translate(LEET)
    return _WS.sub(" ", text).strip()
