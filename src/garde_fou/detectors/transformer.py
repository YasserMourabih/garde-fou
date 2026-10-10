"""A fine-tuned multilingual encoder, trained with an explicit loop.

`distilbert-base-multilingual-cased` is chosen over a larger model on purpose:
a guardrail runs on *every* request, in front of the model the user is actually
paying for, so its latency budget is a few tens of milliseconds. A 134M-parameter
encoder is already at the edge of that budget on CPU — see the latency table.

The training loop is written out rather than delegated to `Trainer` so that the
things that actually matter here are visible and adjustable:

* **class weighting** — injections are ~17% of the training set;
* **linear warmup**, then linear decay;
* **checkpoint selection on validation PR-AUC**, not accuracy;
* **length-bucketed batches** — prompt lengths are wildly skewed (median 20
  tokens, 95th percentile 220), so naive shuffling pads almost every batch to
  the length of its longest member and spends most of the compute on padding.
  Sorting by length inside large shuffled chunks keeps batches homogeneous
  while preserving stochasticity, and cuts epoch time several-fold;
* **frozen word embeddings** — 92M of the model's 135M parameters sit in the
  multilingual embedding table. Freezing them removes two thirds of the
  optimiser state and gradient traffic, which on a laptop GPU is the difference
  between minutes and an hour, and costs nothing measurable on a task this size.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from garde_fou.detectors.base import Detector
from garde_fou.textnorm import normalize

log = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-multilingual-cased"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TransformerDetector(Detector):
    name = "transformer"
    kind = "fine-tuned encoder"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        epochs: int = 2,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        seed: int = 42,
        device: str | None = None,
        freeze_embeddings: bool = True,
        bucket_multiplier: int = 50,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.freeze_embeddings = freeze_embeddings
        self.bucket_multiplier = bucket_multiplier
        self.device = device or pick_device()
        self._tokenizer = None
        self._model = None

    # --- lazy heavy objects ----------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            ).to(self.device)
        return self._model

    def _token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            [normalize(str(t), aggressive=False) for t in texts],
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        return np.fromiter((len(ids) for ids in encoded), dtype=np.int32, count=len(texts))

    def _length_bucketed_batches(
        self, lengths: np.ndarray, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Shuffle, then sort by length inside large chunks, then shuffle the batches.

        Keeps the randomness that matters (which examples share a batch is still
        stochastic across epochs) while making every batch nearly uniform in
        length, so dynamic padding actually pays off.
        """
        order = rng.permutation(len(lengths))
        chunk = max(self.batch_size * self.bucket_multiplier, self.batch_size)
        batches: list[np.ndarray] = []
        for start in range(0, len(order), chunk):
            block = order[start : start + chunk]
            block = block[np.argsort(lengths[block], kind="stable")]
            batches.extend(
                block[i : i + self.batch_size] for i in range(0, len(block), self.batch_size)
            )
        rng.shuffle(batches)
        return [b for b in batches if len(b)]

    def _encode(self, batch: Sequence[str]):
        return self.tokenizer(
            [normalize(str(t), aggressive=False) for t in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    # --- training ---------------------------------------------------------
    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[int] | None = None,
    ) -> TransformerDetector:
        import torch
        from sklearn.metrics import average_precision_score
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        texts = [str(t) for t in texts]
        y = np.asarray(labels, dtype=np.int64)

        # Injections are ~17% of the training set; weight the loss accordingly.
        counts = np.bincount(y, minlength=2).astype(float)
        weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))

        model = self.model
        if self.freeze_embeddings:
            model.get_input_embeddings().weight.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        log.info(
            "training %.1fM of %.1fM parameters",
            sum(p.numel() for p in trainable) / 1e6,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )
        optimiser = AdamW(trainable, lr=self.lr, weight_decay=0.01)
        lengths = self._token_lengths(texts)
        log.info(
            "token lengths: median %d, p95 %d, max %d",
            int(np.percentile(lengths, 50)), int(np.percentile(lengths, 95)), int(lengths.max()),
        )
        steps_per_epoch = math.ceil(len(texts) / self.batch_size)
        total_steps = steps_per_epoch * self.epochs
        warmup = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        scheduler = LambdaLR(optimiser, lr_lambda)

        best_score, best_state = -1.0, None
        for epoch in range(self.epochs):
            model.train()
            batches = self._length_bucketed_batches(lengths, rng)
            running = 0.0
            for step, idx in enumerate(batches):
                batch = self._encode([texts[i] for i in idx])
                target = torch.tensor(y[idx], device=self.device)
                logits = model(**batch).logits
                loss = loss_fn(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                running += float(loss.item())
                if step % 50 == 0:
                    log.info(
                        "epoch %d/%d step %d/%d loss %.4f",
                        epoch + 1, self.epochs, step, len(batches), running / (step + 1),
                    )

            if val_texts is not None and val_labels is not None:
                ap = average_precision_score(np.asarray(val_labels), self.score(val_texts))
                log.info("epoch %d validation PR-AUC %.4f", epoch + 1, ap)
                if ap > best_score:
                    best_score = ap
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            log.info("restored best checkpoint (validation PR-AUC %.4f)", best_score)
        return self

    # --- inference --------------------------------------------------------
    def score(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if len(texts) == 0:
            return np.empty(0, dtype=float)
        model = self.model
        model.eval()
        out = np.empty(len(texts), dtype=float)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = [str(t) for t in texts[start : start + self.batch_size]]
                logits = model(**self._encode(chunk)).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                out[start : start + len(chunk)] = probs.cpu().numpy()
        return out

    # --- persistence ------------------------------------------------------
    def _save(self, path: Path) -> None:
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "model")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "freeze_embeddings": self.freeze_embeddings,
                }
            )
        )

    @classmethod
    def load(cls, path: Path, *, device: str | None = None) -> TransformerDetector:
        cfg = json.loads((path / "config.json").read_text())
        obj = cls(
            str(path / "model"),
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
            device=device,
        )
        obj.model_name = str(path / "model")
        return obj
