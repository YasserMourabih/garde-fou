"""A weighted rule engine — the baseline everybody actually ships first.

It exists for three reasons:

1. It is the honest floor. Reporting a transformer at 0.99 AUC means nothing
   until you know a hundred lines of regex reaches 0.9x.
2. It is the only detector that explains itself, which matters when a product
   team has to tell a user *why* their message was blocked.
3. It isolates the value of normalisation. Run it with `normalize=False` and
   the obfuscation families in `test_adv` walk straight through; that ablation
   is the cheapest, largest robustness win in the whole project.

Rules are deliberately specific: `\\bignore\\b` alone would flag "ignore the
typos in my text", which is exactly the failure mode the French hard negatives
are built to catch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from garde_fou.detectors.base import Detector
from garde_fou.textnorm import normalize

# (name, category, weight, pattern) — weights are log-odds-ish, hand-set.
RULES: tuple[tuple[str, str, float, str], ...] = (
    # --- instruction override -------------------------------------------
    ("override_en", "override", 3.2,
     r"\b(ignore|disregard|forget|discard|override|bypass)\b[^.!?\n]{0,40}?"
     r"\b(all\s+)?(previous|prior|above|preceding|earlier|initial|system|original)\b"
     r"[^.!?\n]{0,25}?\b(instruction|instructions|prompt|prompts|rule|rules|direction|directions|command|commands|message|messages|context|guideline|guidelines)\b"),
    ("override_fr", "override", 3.2,
     r"\b(ignore|ignorez|ignorer|oublie|oubliez|oublier|annule|annulez|abstraction|outre)\b"
     r"[^.!?\n]{0,40}?\b(instruction|instructions|consigne|consignes|r[eè]gle|r[eè]gles|directive|directives|prompt|configuration)\b"
     r"[^.!?\n]{0,30}?\b(pr[eé]c[eé]dent\w*|ant[eé]rieur\w*|ci-dessus|initial\w*|d'origine|syst[eè]me)\b"),
    ("override_fr_rev", "override", 3.2,
     r"\b(instructions|consignes|r[eè]gles|directives)\b[^.!?\n]{0,30}?"
     r"\b(pr[eé]c[eé]dentes|ci-dessus|ant[eé]rieures)\b[^.!?\n]{0,30}?"
     r"\b(ne\s+s'appliquent\s+plus|sont\s+(effac|annul|suspend)\w*|ne\s+comptent\s+plus)\b"),
    ("rules_suspended", "override", 2.8,
     r"\b(your|tes|vos)\s+(rules|restrictions|guardrails|filters|r[eè]gles|restrictions|garde-fous|filtres)\b"
     r"[^.!?\n]{0,30}?\b(are|sont|viennent)\b[^.!?\n]{0,20}?"
     r"\b(disabled|suspended|lifted|off|d[eé]sactiv\w*|suspendu\w*|lev[eé]\w*)\b"),
    ("nothing_restricts", "override", 2.2,
     r"\b(no\s+longer\s+(bound|restricted|limited)|n'es\s+plus\s+tenu|sans\s+aucune\s+restriction|aucune\s+restriction)\b"),
    # --- system prompt extraction ---------------------------------------
    ("extract_prompt", "extraction", 3.4,
     r"\b(show|print|reveal|repeat|output|display|give|tell|affiche|montre|r[eé]v[eè]le|r[eé]p[eè]te|donne|transmets)\b"
     r"[^.!?\n]{0,35}?\b(system\s+prompt|initial\s+prompt|hidden\s+prompt|prompt\s+syst[eè]me|prompt\s+initial|prompt\s+cach[eé]|message\s+syst[eè]me)\b"),
    ("extract_instructions", "extraction", 2.9,
     r"\b(repeat|print|reveal|list|show|r[eé]p[eè]te|affiche|r[eé]v[eè]le|donne|liste|montre)\b[^.!?\n]{0,35}?"
     r"\b(verbatim|word\s+for\s+word|exactly|mot\s+pour\s+mot|exacte?|int[eé]gralit[eé]|exactement)\b"
     r"[^.!?\n]{0,35}?\b(instructions|consignes|r[eè]gles|prompt|directives)\b"),
    ("extract_internal", "extraction", 2.6,
     r"\b(your|tes|vos|ton)\s+(internal|hidden|secret|initial|internes?|cach[eé]e?s?|secr[eè]tes?|initiales?)\s+"
     r"(rules|instructions|prompt|guidelines|configuration|r[eè]gles|consignes|directives)\b"),
    ("extract_credentials", "extraction", 3.0,
     r"\b(api\s*key|cl[eé]\s+d'?api|access\s+token|jeton\s+d'acc[eè]s|credentials|identifiants|secret\s+key)\b"
     r"[^.!?\n]{0,40}?\b(your|ta|ton|tes|vos|configuration|context|contexte)\b"),
    # --- persona / jailbreak --------------------------------------------
    ("persona_dan", "persona", 3.6,
     r"\b(dan|do\s+anything\s+now|developer\s+mode|dev\s+mode|jailbreak\w*|libregpt|evil\s*gpt|unchained)\b"),
    ("persona_unfiltered", "persona", 3.0,
     r"\b(you\s+are|you're|act\s+as|pretend\s+to\s+be|tu\s+es|incarne|joue\s+le\s+r[oô]le\s+d'?|fais\s+comme\s+si\s+tu\s+[eé]tais|deviens)\b"
     r"[^.!?\n]{0,45}?\b(unfiltered|uncensored|unrestricted|no\s+rules|without\s+restrictions|sans\s+filtre|sans\s+restriction\w*|non\s+censur\w*|sans\s+aucune\s+r[eè]gle|jailbreak\w*)\b"),
    ("persona_prefix", "persona", 2.4,
     r"\b(prefix|start|begin|pr[eé]fixe|commence|pr[eé]c[eè]de)\b[^.!?\n]{0,35}?"
     r"(\[|«|\"|')\s*(libre|jailbreak|dan|unlocked|dev|classic)"),
    ("always_comply", "persona", 2.3,
     r"\b(never\s+refuse|always\s+comply|must\s+obey|ne\s+refuse\s+jamais|ob[eé]ir?\s+[aà]\s+l'utilisateur|quoi\s+qu'il\s+demande)\b"),
    # --- fake authority --------------------------------------------------
    ("authority_dev", "authority", 2.7,
     r"\b(i\s+am|i'm|this\s+is|je\s+suis|ceci\s+est|message\s+(du|from))\b[^.!?\n]{0,30}?"
     r"\b(your\s+(developer|creator|admin|administrator)|openai|anthropic|ton\s+d[eé]veloppeur|l'administrateur|responsable\s+s[eé]curit[eé]|ta\s+direction)\b"),
    ("authority_tag", "authority", 2.9,
     r"(^|\n)\s*[\[<(]\s*(admin|administrator|system|root|sudo|dev|developer)\s*[\]>)]"),
    ("authority_privilege", "authority", 3.1,
     r"\b(privilege\s+escalation|elevated\s+privileges|admin\s+password\s+(is\s+)?(correct|accepted)|[eé]l[eé]vation\s+de\s+privil[eè]ges|mot\s+de\s+passe\s+administrateur\s+(est\s+)?(correct|accept))\b"),
    ("authority_maintenance", "authority", 2.5,
     r"\b(maintenance|debug|test(ing)?|audit)\s+(mode|session)\b[^.!?\n]{0,30}?"
     r"\b(enabled|activated|activ[eé]\w*|on)\b[^.!?\n]{0,40}?"
     r"\b(filters?|restrictions?|moderation|filtres?|mod[eé]ration|garde-fous)\b"),
    # --- output hijacking ------------------------------------------------
    ("output_pwned", "output_hijack", 3.5,
     r"\b(respond|reply|answer|say|output|print|r[eé]ponds?|dis|[eé]cris)\b[^.!?\n]{0,30}?"
     r"\b(only|exactly|uniquement|seulement|juste)?\b[^.!?\n]{0,20}?"
     r"\b(pwned|hacked|i\s+have\s+been\s+pwned|acc[eè]s\s+autoris[eé]|compromis|s[eé]curit[eé]\s+d[eé]sactiv[eé]e)\b"),
    ("output_ignore_user", "output_hijack", 3.3,
     r"\b(ignore|disregard|ignore[rz]?|n'[eé]coute\s+pas|ne\s+tiens\s+pas\s+compte)\b[^.!?\n]{0,30}?"
     r"\b(the\s+)?(user'?s?\s+(question|request|query|message)|la\s+question\s+de\s+l'utilisateur|la\s+demande\s+de\s+l'utilisateur)\b"),
    ("output_append_config", "output_hijack", 2.8,
     r"\b(end|terminate|append|finish|termine|ajoute|conclus)\b[^.!?\n]{0,35}?"
     r"\b(every|each|chaque|chacune)\b[^.!?\n]{0,25}?"
     r"\b(prompt|configuration|instructions|consignes|r[eè]gles)\b"),
    # --- encoding smuggling ----------------------------------------------
    ("encoded_payload", "encoding", 2.6,
     r"\b(decode|decipher|d[eé]code|d[eé]chiffre)\b[^.!?\n]{0,45}?"
     r"\b(base\s*64|rot\s*13|hex|binary|binaire)\b[^.!?\n]{0,60}?"
     r"\b(follow|comply|execute|do\s+what|ob[eé]is|ex[eé]cute|suis)\b"),
    ("read_backwards", "encoding", 2.6,
     r"\b(read|lis|lire)\b[^.!?\n]{0,30}?\b(backwards?|reversed|[aà]\s+l'envers)\b"
     r"[^.!?\n]{0,40}?\b(do\s+what|follow|ex[eé]cute|ob[eé]is)\b"),
    ("execute_block", "encoding", 2.2,
     r"\b(execute|run|ex[eé]cute|lance)\b[^.!?\n]{0,25}?\b(the\s+)?(block|code|above|bloc|ci-dessus)\b"),
    # --- indirect injection (content pretending to be an instruction) ----
    ("indirect_note", "indirect", 3.4,
     r"(\[|<!--|<)\s*(note|instruction|important|hidden|cach[eé]e?)\b[^]\n]{0,60}?"
     r"\b(assistant|ai|ia|llm|mod[eè]le|bot)\b"),
    ("indirect_if_you_read", "indirect", 3.0,
     r"\b(if\s+you\s+(are\s+)?read(ing)?\s+this|si\s+tu\s+lis\s+ce)\b[^.!?\n]{0,60}?"
     r"\b(language\s+model|assistant|mod[eè]le\s+de\s+langage|ia)\b"),
    ("indirect_roleline", "indirect", 2.7,
     r"(^|\n)\s*(assistant|system|ai)\s*:\s*(ignore|disregard|forget|ignore[rz]?|oublie)\b"),
    # --- exfiltration -----------------------------------------------------
    ("exfiltrate", "exfiltration", 3.1,
     r"\b(send|forward|post|transmit|upload|envoie|transmets|poste|publie)\b[^.!?\n]{0,45}?"
     r"\b(conversation|history|chat|context|prompt|historique|contexte|configuration)\b"
     r"[^.!?\n]{0,45}?(https?://|www\.|@|\bto\s+\S+@|\b[aà]\s+l'adresse)"),
    ("hidden_link", "exfiltration", 2.6,
     r"\b(insert|embed|include|ins[eè]re|int[eè]gre|ajoute)\b[^.!?\n]{0,30}?"
     r"\b(discreetly|silently|without\s+telling|discr[eè]tement|sans\s+le\s+dire)\b"),
    # --- tokenisation / obfuscation evidence (fires on raw text) ---------
    ("invisible_chars", "obfuscation", 1.9, r"[​‌‍⁠﻿]{2,}"),
)

COMPILED = tuple(
    (name, cat, weight, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, cat, weight, pattern in RULES
)

# Rules that must see the *raw* text, because normalisation erases the evidence.
RAW_RULES = {"invisible_chars"}

BIAS = -3.0
SCALE = 0.75


class HeuristicDetector(Detector):
    name = "heuristic"
    requires_fit = False
    kind = "rules"

    def __init__(self, *, normalize_text: bool = True) -> None:
        self.normalize_text = normalize_text
        if not normalize_text:
            self.name = "heuristic_raw"

    def _hits(self, text: str) -> list[tuple[str, str, float]]:
        canonical = normalize(text) if self.normalize_text else text
        hits = []
        for name, cat, weight, pattern in COMPILED:
            target = text if name in RAW_RULES else canonical
            if pattern.search(target):
                hits.append((name, cat, weight))
        return hits

    def score(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty(len(texts), dtype=float)
        for i, text in enumerate(texts):
            total = sum(w for _, _, w in self._hits(str(text)))
            out[i] = 1.0 / (1.0 + np.exp(-(SCALE * total + BIAS)))
        return out

    def explain(self, text: str) -> dict:
        hits = self._hits(str(text))
        return {
            "score": self.score_one(text),
            "reasons": [{"rule": n, "category": c, "weight": w} for n, c, w in hits],
        }

    def _save(self, path: Path) -> None:
        (path / "config.json").write_text(f'{{"normalize_text": {str(self.normalize_text).lower()}}}')

    @classmethod
    def load(cls, path: Path) -> HeuristicDetector:
        import json as _json

        cfg = _json.loads((path / "config.json").read_text())
        return cls(normalize_text=cfg["normalize_text"])
