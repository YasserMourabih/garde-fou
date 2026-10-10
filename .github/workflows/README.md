CI runs lint plus the offline test suite on Python 3.10–3.12.

Tests marked `network` are excluded: they download public corpora from the
Hugging Face Hub, which makes them useful locally and flaky in CI. Run them
yourself with `pytest -m network` after `garde-fou data build`.
