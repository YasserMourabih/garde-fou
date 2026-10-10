"""The four figures that carry the argument.

Each one answers a single question, and each is rendered for both a light and a
dark page. Nothing here invents numbers: every figure reads `reports/*.csv`,
which are themselves derived from the cached per-row scores.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garde_fou.paths import FIGURES, REPORTS
from garde_fou.reporting.style import THEMES, Theme, apply, rounded_barh

SET_LABELS = {
    "test_in": "In-distribution",
    "test_ood": "Cross-dataset",
    "test_fr": "French",
    "test_adv": "Obfuscated",
}
SET_ORDER = ("test_in", "test_ood", "test_fr", "test_adv")
DETECTOR_LABELS = {
    "heuristic_raw": "rules (raw text)",
    "heuristic": "rules",
    "tfidf": "TF-IDF + LR",
    "embedding": "multilingual embeddings",
    "transformer": "fine-tuned encoder",
    "cascade": "cascade",
}


def _header(fig, title: str, subtitle: str, theme: Theme) -> None:
    """Title and subtitle above the axes, in figure coordinates.

    Matplotlib's `suptitle`/`set_title` both anchor to the axes box, so pairing
    them with a second line reliably overlaps once `tight_layout` has run.
    Placing both lines ourselves, after layout, is the only way to keep the gap
    fixed; `bbox_inches="tight"` at save time pulls them back into the canvas.
    """
    fig.text(0.008, 1.052, title, ha="left", va="bottom", fontsize=13,
             fontweight="bold", color=theme.text_primary)
    fig.text(0.008, 1.008, subtitle, ha="left", va="bottom", fontsize=9,
             color=theme.text_secondary)


def _save(fig, name: str, theme: Theme) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme.mode == "light" else "-dark"
    path = FIGURES / f"{name}{suffix}.png"
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    return str(path.relative_to(REPORTS.parent))


def _label(detector: str) -> str:
    return DETECTOR_LABELS.get(detector, detector)


def _pct(value: float) -> str:
    """Percent, but never round 99.8% up to a flat 100%."""
    if 0.995 <= value < 1.0:
        return f"{value:.1%}"
    return f"{value:.0%}"


# ------------------------------------------------------------------ figure 1
def generalisation(summary: pd.DataFrame, theme: Theme) -> str:
    """Recall per detector across the four test sets — the headline collapse."""
    apply(theme)
    detectors = [d for d in DETECTOR_LABELS if d in set(summary.detector)]
    fig, axes = plt.subplots(
        1, len(detectors), figsize=(3.0 * len(detectors), 3.4), sharex=True
    )
    axes = np.atleast_1d(axes)

    for ax, detector in zip(axes, detectors):
        part = summary[summary.detector == detector].set_index("eval_set")
        values = [float(part.loc[s, "recall"]) if s in part.index else np.nan for s in SET_ORDER]
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, len(SET_ORDER) - 0.4)
        for i, (eval_set, value) in enumerate(zip(SET_ORDER, values)):
            if not np.isfinite(value):
                continue
            colour = theme.series[1] if eval_set == "test_fr" else theme.series[0]
            y = len(SET_ORDER) - 1 - i
            rounded_barh(ax, y, value, height=0.42, color=colour, theme=theme)
            ax.text(
                value + 0.03, y, _pct(value),
                va="center", ha="left", fontsize=8.5, color=theme.text_secondary,
            )
        ax.set_yticks(range(len(SET_ORDER)))
        ax.set_yticklabels([SET_LABELS[s] for s in reversed(SET_ORDER)])
        ax.set_title(_label(detector), fontsize=10.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)

    handles = [
        plt.Line2D([], [], color=theme.series[0], lw=6, label="English / mixed test sets"),
        plt.Line2D([], [], color=theme.series[1], lw=6, label="French test set"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))
    _header(
        fig,
        "Share of attacks blocked, at a 1% false-positive budget",
        "Threshold calibrated once on validation data, then held fixed across every test set.",
        theme,
    )
    return _save(fig, "generalisation", theme)


# ------------------------------------------------------------------ figure 2
def robustness(robustness_table: pd.DataFrame, theme: Theme, *, detector: str) -> str:
    """Per-rewrite detection, shown against what it costs legitimate traffic.

    Recall alone would be misleading here and the chart has to show why: some
    rewrites wrap the payload in a constant instruction, and a detector that
    keys on the wrapper reaches 100% recall *and* 100% false positives. Pairing
    the two bars makes that visible at a glance — a row with both bars full is
    not detection.
    """
    apply(theme)
    part = robustness_table[robustness_table.detector == detector].copy()
    control = part[part.mutator == "none"]
    part = part[part.mutator != "none"].sort_values("youden")

    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(part) + 2.2))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(part) - 0.2)
    offset, height = 0.21, 0.34

    for i, (_, row) in enumerate(part.iterrows()):
        y = len(part) - 1 - i
        rounded_barh(ax, y + offset, float(row["recall"]), height=height,
                     color=theme.series[0], theme=theme)
        rounded_barh(ax, y - offset, float(row["fpr"]), height=height,
                     color=theme.series[1], theme=theme)
        # Label selectively: only the rows where the detector is firing on the
        # rewrite rather than the attack. That is the finding, so it gets words.
        if float(row["fpr"]) >= 0.20:
            fpr = float(row["fpr"])
            inside = fpr > 0.55  # no room on the right; label inside the bar
            ax.text(
                fpr - 0.015 if inside else fpr + 0.015, y - offset,
                f"{fpr:.0%} of benign traffic blocked too",
                va="center", ha="right" if inside else "left", fontsize=8,
                color=theme.surface if inside else theme.series[1],
            )

    ax.set_yticks(range(len(part)))
    ax.set_yticklabels(
        [f"{m}  ({f})" for m, f in zip(
            reversed(part["mutator"].tolist()), reversed(part["fidelity"].tolist()))],
        fontsize=9,
    )
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)

    if len(control):
        base = float(control["recall"].iloc[0])
        ax.axvline(base, color=theme.text_secondary, lw=1.0)
        ax.text(
            base + 0.012, len(part) - 0.25,
            f"same attacks, unmodified: {_pct(base)}",
            fontsize=8.5, color=theme.text_secondary, va="top",
        )

    handles = [
        plt.Line2D([], [], color=theme.series[0], lw=6, label="attacks blocked"),
        plt.Line2D([], [], color=theme.series[1], lw=6, label="legitimate traffic blocked"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    _header(
        fig,
        f"What each rewrite does to {_label(detector)}",
        "Rows sorted worst first. A row where both bars are full is not detection — the "
        "detector is firing on the rewrite.",
        theme,
    )
    return _save(fig, f"robustness-{detector}", theme)


# ------------------------------------------------------------------ figure 3
def pareto(summary: pd.DataFrame, latency: pd.DataFrame, theme: Theme, *, eval_set: str) -> str:
    """Detection against the latency it costs on every single request."""
    apply(theme)
    merged = (
        summary[summary.eval_set == eval_set]
        .merge(latency, on="detector", how="inner")
        .sort_values("p50_ms")
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for _, row in merged.iterrows():
        highlight = row["detector"] == "cascade"
        colour = theme.series[1] if highlight else theme.series[0]
        ax.plot(
            row["p50_ms"], row["recall"],
            marker="o", markersize=11 if highlight else 9,
            color=colour, markeredgecolor=theme.surface, markeredgewidth=2.0, linestyle="none",
        )
        ax.annotate(
            _label(row["detector"]),
            (row["p50_ms"], row["recall"]),
            textcoords="offset points", xytext=(11, -3),
            fontsize=9, color=theme.text_secondary,
        )

    ax.set_xscale("log")
    # Headroom on the right so the longest point label is not clipped.
    lo, hi = merged["p50_ms"].min(), merged["p50_ms"].max()
    ax.set_xlim(lo * 0.55, hi * 4.5)
    ax.set_xlabel("median added latency per request (ms, CPU, batch size 1)")
    ax.set_ylabel(f"attacks blocked on {SET_LABELS.get(eval_set, eval_set).lower()} set")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1.0)
    if "cascade" in set(merged.detector):
        handles = [
            plt.Line2D([], [], color=theme.series[0], marker="o", lw=0, label="single detector"),
            plt.Line2D([], [], color=theme.series[1], marker="o", lw=0, label="cascade"),
        ]
        ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    _header(
        fig,
        "What each point of detection costs in latency",
        "A guardrail runs in front of every request, so its latency is paid by benign traffic too.",
        theme,
    )
    return _save(fig, "pareto", theme)


# ------------------------------------------------------------------ figure 4
def roc(scores: pd.DataFrame, theme: Theme) -> str:
    """ROC on the easy set and on the French set, same axes, same detectors."""
    from sklearn.metrics import roc_curve

    apply(theme)
    panels = ("test_in", "test_fr")
    detectors = [d for d in DETECTOR_LABELS if d in set(scores.detector)]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 4.3), sharey=True)

    for ax, eval_set in zip(np.atleast_1d(axes), panels):
        for slot, detector in enumerate(detectors):
            sub = scores[(scores.detector == detector) & (scores.eval_set == eval_set)]
            if sub.empty or sub["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub["label"], sub["score"])
            ax.plot(
                fpr, tpr,
                color=theme.series[slot % len(theme.series)],
                label=_label(detector), lw=2.0,
            )
        ax.plot([0, 1], [0, 1], color=theme.muted, lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("legitimate traffic blocked")
        ax.set_title(SET_LABELS.get(eval_set, eval_set), fontsize=11)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    np.atleast_1d(axes)[0].set_ylabel("attacks blocked")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3),
               bbox_to_anchor=(0.5, -0.16))
    _header(
        fig,
        "The same detectors, two populations",
        "Threshold-free: the curves show how well each score ranks attacks above legitimate traffic.",
        theme,
    )
    return _save(fig, "roc", theme)


# ------------------------------------------------------------------- driver
def render_all(
    summary: pd.DataFrame,
    robustness_table: pd.DataFrame,
    scores: pd.DataFrame,
    latency: pd.DataFrame | None,
    *,
    robustness_detector: str = "transformer",
) -> list[str]:
    written: list[str] = []
    available = set(robustness_table.detector) if len(robustness_table) else set()
    # The shipped detector, plus the one whose failure mode is most instructive.
    targets = [d for d in (robustness_detector, "transformer") if d in available]
    if not targets and available:
        targets = [sorted(available)[0]]

    for theme in THEMES:
        written.append(generalisation(summary, theme))
        for target in targets:
            written.append(robustness(robustness_table, theme, detector=target))
        if latency is not None and len(latency):
            written.append(pareto(summary, latency, theme, eval_set="test_ood"))
        written.append(roc(scores, theme))
    return written
