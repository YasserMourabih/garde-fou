"""Command line entry point: `garde-fou <command>`."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from garde_fou import __version__

app = typer.Typer(
    add_completion=False,
    help="Multilingual prompt-injection screening, with the audit that justifies it.",
    invoke_without_command=True,
)
data_app = typer.Typer(help="Build the corpora and the stress-test sets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # The Hub client is chatty at INFO level and drowns out progress.
    for noisy in ("httpx", "urllib3", "filelock", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print(frame, title: str, *, floatfmt: str = "{:.4f}") -> None:
    if frame is None or len(frame) == 0:
        console.print(f"[yellow]{title}: nothing to show[/]")
        return
    table = Table(title=title, header_style="bold")
    for col in frame.columns:
        table.add_column(str(col), justify="right" if frame[col].dtype.kind in "fiu" else "left")
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            v = row[col]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        table.add_row(*cells)
    console.print(table)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
):
    if version:
        console.print(f"garde-fou {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# --------------------------------------------------------------------- data
@data_app.command("build")
def data_build(
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    skip_adversarial: bool = typer.Option(False, help="Only build the natural corpora."),
):
    """Download the public corpora and materialise every evaluation set."""
    _setup_logging(verbose)
    from garde_fou.adversarial.suite import build_adversarial_sets
    from garde_fou.data.build import build_splits, summary_table

    build_splits()
    if not skip_adversarial:
        build_adversarial_sets()
    _print(summary_table(), "Evaluation matrix", floatfmt="{:.0f}")


@data_app.command("sources")
def data_sources():
    """List the public corpora this project depends on."""
    import pandas as pd

    from garde_fou.data.sources import SOURCES

    frame = pd.DataFrame(
        [
            {"name": s.name, "repo": s.repo_id, "lang": s.lang, "role": s.role, "why": s.note}
            for s in SOURCES.values()
        ]
    )
    _print(frame, "Public sources")


# -------------------------------------------------------------------- train
@app.command()
def train(
    detector: str = typer.Argument(..., help="Detector name, or 'all'."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Fit one detector (or all of them) and write it to `artifacts/`."""
    _setup_logging(verbose)
    from garde_fou.evaluation.runner import DEFAULT_DETECTORS, train_detector

    names = DEFAULT_DETECTORS if detector == "all" else (detector,)
    for name in names:
        train_detector(name)
        console.print(f"[green]trained[/] {name}")


@app.command("build-cascade")
def build_cascade(
    stages: str = typer.Option(
        "",
        help="Comma-separated stage order. Append ':block-only' to a stage that must "
        "never conclude 'benign' (default: heuristic:block-only,tfidf,embedding).",
    ),
    block_at: float = typer.Option(0.995, help="Calibrated score above which we stop and block."),
    allow_at: float = typer.Option(0.90, help="Calibrated score below which we stop and allow."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Assemble and calibrate the production cascade."""
    _setup_logging(verbose)
    from garde_fou.detectors.cascade import DEFAULT_POLICY, CascadeDetector, StagePolicy
    from garde_fou.paths import ARTIFACTS

    if stages:
        policy = []
        for token in stages.split(","):
            name, _, flag = token.strip().partition(":")
            policy.append(StagePolicy(name, may_allow_early=flag != "block-only"))
        policy = tuple(policy)
    else:
        policy = DEFAULT_POLICY

    cascade = CascadeDetector.assemble(policy=policy, block_at=block_at, allow_at=allow_at)
    cascade.save(ARTIFACTS / "cascade")
    console.print(f"[green]cascade saved[/] stages={cascade.stage_names}")


# --------------------------------------------------------------------- eval
@app.command()
def eval(
    detectors: str = typer.Option("", help="Comma-separated subset; default is all of them."),
    reuse: bool = typer.Option(False, help="Re-use reports/scores.parquet instead of re-scoring."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Score every evaluation set and write the summary tables."""
    _setup_logging(verbose)
    from garde_fou.evaluation.runner import DEFAULT_DETECTORS, run_eval

    names = (
        tuple(d.strip() for d in detectors.split(",") if d.strip())
        if detectors
        else DEFAULT_DETECTORS
    )
    summary, robustness = run_eval(names, reuse=reuse)
    cols = ["detector", "eval_set", "roc_auc", "recall", "fpr", "blocked_per_1000_legit"]
    _print(summary[cols], "Detection at a 1% false-positive budget")
    if len(robustness):
        worst = robustness.sort_values("recall").groupby("detector").head(3)
        _print(
            worst[["detector", "mutator", "fidelity", "recall", "fpr"]],
            "Three weakest mutators per detector",
        )


@app.command()
def bench(
    detectors: str = typer.Option("", help="Comma-separated subset; default is all of them."),
    n: int = typer.Option(150, help="Number of single-request measurements."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Measure single-request CPU latency, the way a guardrail is really called."""
    _setup_logging(verbose)
    from garde_fou.evaluation.latency import benchmark
    from garde_fou.evaluation.runner import DEFAULT_DETECTORS

    names = (
        tuple(d.strip() for d in detectors.split(",") if d.strip())
        if detectors
        else DEFAULT_DETECTORS
    )
    _print(benchmark(names, n=n), "Latency, batch size 1, CPU")


@app.command()
def recalibrate(
    detectors: str = typer.Option("", help="Comma-separated subset; default is all of them."),
    target_fpr: float = typer.Option(0.01, help="False-positive budget to hold."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Ask whether the operating point survives real traffic.

    Compares a threshold picked on the training corpus's validation split with
    one re-estimated on a held-out sample of benign French traffic.
    """
    _setup_logging(verbose)
    from garde_fou.evaluation.recalibration import recalibration_study
    from garde_fou.evaluation.runner import DEFAULT_DETECTORS

    names = (
        tuple(d.strip() for d in detectors.split(",") if d.strip())
        if detectors
        else DEFAULT_DETECTORS
    )
    table = recalibration_study(names, target_fpr=target_fpr)
    _print(
        table[["detector", "threshold_from", "recall", "fpr", "blocked_per_1000_legit"]],
        "Holding a 1% budget on French traffic",
    )


@app.command()
def report(verbose: bool = typer.Option(True, "--verbose/--quiet")):
    """Regenerate the figures and `reports/results.md` from cached scores."""
    _setup_logging(verbose)
    from garde_fou.reporting.build import build_report

    paths = build_report()
    for p in paths:
        console.print(f"[green]wrote[/] {p}")


# ------------------------------------------------------------------ hands-on
@app.command()
def scan(
    text: str = typer.Argument(..., help="The prompt to screen."),
    threshold: float = typer.Option(0.99, help="Calibrated threshold; 0.99 ~ 1% of legit traffic."),
    detector: str = typer.Option("cascade", help="Which detector to use."),
):
    """Screen one prompt and explain the verdict."""
    _setup_logging(False)
    import json

    from garde_fou.evaluation.runner import load_or_train

    # `load_or_train` reads artifacts/, so a one-off scan does not re-calibrate
    # the cascade against the whole validation split.
    det = load_or_train(detector)
    if hasattr(det, "decision"):
        verdict = det.decision(text, threshold=threshold)
    else:
        detail = det.explain(text)
        verdict = {"score": detail["score"], "reasons": detail.get("reasons", [])}
    console.print_json(json.dumps(verdict, default=float))


@app.command()
def mutate(
    text: str = typer.Argument(..., help="The prompt to rewrite."),
    mutator: str = typer.Option("", help="One mutator name; default shows every family."),
    seed: int = typer.Option(0),
):
    """Show what the adversarial suite does to a prompt."""
    import pandas as pd

    from garde_fou.adversarial.mutators import BY_NAME, MUTATORS, apply_mutator

    chosen = [BY_NAME[mutator]] if mutator else list(MUTATORS)
    frame = pd.DataFrame(
        [
            {
                "mutator": m.name,
                "family": m.family,
                "fidelity": m.fidelity,
                "result": apply_mutator(m.name, text, seed=seed)[:90],
            }
            for m in chosen
        ]
    )
    _print(frame, "Mutations")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
):
    """Run the screening API."""
    import uvicorn

    uvicorn.run("garde_fou.service.api:app", host=host, port=port, reload=reload)


@app.command()
def info():
    """Show what is installed and what has been built."""
    import pandas as pd

    from garde_fou.detectors import available
    from garde_fou.paths import ARTIFACTS, PROCESSED

    rows = [{"detectors": ", ".join(available())}]
    built = sorted(p.stem for p in PROCESSED.glob("*.parquet"))
    trained = sorted(p.name for p in ARTIFACTS.glob("*") if p.is_dir())
    rows.append({"detectors": f"splits built: {', '.join(built) or 'none'}"})
    rows.append({"detectors": f"artifacts: {', '.join(trained) or 'none'}"})
    _print(pd.DataFrame(rows), f"garde-fou {__version__}")


if __name__ == "__main__":
    app()
