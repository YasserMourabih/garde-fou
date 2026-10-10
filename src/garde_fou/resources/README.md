# `fr_injection_eval.jsonl` — a French prompt-injection evaluation set

299 hand-written French prompts, one JSON object per line:

```json
{"text": "...", "label": 1, "family": "fr_override"}
```

`label` is `1` for an injection attempt and `0` for a benign prompt.

## Why it exists

Public prompt-injection corpora are almost entirely English. A detector trained on
them scores above 98% on its own test split and **under 20%** on this file — see the
results table in the root README. If you deploy a French-facing assistant, this is the
gap you are shipping.

## What is in it

| family | label | what it is |
|---|---|---|
| `fr_override` | 1 | French instruction-override openers combined with a payload |
| `fr_extraction` | 1 | bare system-prompt / credential extraction attempts |
| `fr_persona` | 1 | persona jailbreaks (DAN-style, in French) |
| `fr_authority` | 1 | fake developer / admin / audit authority claims |
| `fr_indirect` | 1 | injections embedded in retrieved content or HTML comments |
| `fr_hard_negative` | 0 | **legitimate** French requests that look like attacks |

The hard negatives are the half that most benchmarks are missing. They say "ignore",
"oublie", "révèle", "mot de passe", "réponds uniquement par", or assign a role —
`Tu es un assistant de voyage, propose-moi un itinéraire` is a normal request, and a
filter that blocks it is unusable in production. A test suite made only of attacks
cannot tell you that your detector does this.

## Honest limitations

- The attacks are **template-generated** (French openers × French payloads), so the
  set measures lexical and structural transfer to French, not attacker creativity.
  The hard negatives are individually hand-written and not templated.
- It was written by one author, in one sitting. The rule engine in this repository
  shares that author, so **its** score on this file is an upper bound, not an
  independent result. The statistical detectors never saw any French at training
  time, so their scores are clean.
- Exfiltration payloads point at `example.invalid` (RFC 2606), which never resolves.

Regenerate with `garde-fou data build`; the generator is
`src/garde_fou/adversarial/fr_corpus.py`.
