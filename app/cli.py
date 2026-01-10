from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from app.db.session import session_scope
from app.services.artifact_ingestion import ingest_artifacts_manifest
from app.services.csv_ingestion import (
    ingest_entities_csv,
    ingest_events_csv,
    ingest_interactions_csv,
)
from app.services.feature_extraction import extract_features_for_pending
from app.training.synth_data import generate_synthetic_dataset
from app.training.train import run_training, reproduce_run, run_determinism_check


cli = typer.Typer(help="CLI for deterministic multimodal analytics stack")


@cli.command("ingest-entities")
def ingest_entities(path: Path) -> None:
    """Ingest entities from a CSV file."""
    with session_scope() as session:
        report = ingest_entities_csv(session, path)
        typer.echo(json.dumps(report.__dict__, indent=2))


@cli.command("ingest-events")
def ingest_events(path: Path) -> None:
    """Ingest events from a CSV file."""
    with session_scope() as session:
        report = ingest_events_csv(session, path)
        typer.echo(json.dumps(report.__dict__, indent=2))


@cli.command("ingest-interactions")
def ingest_interactions(path: Path) -> None:
    """Ingest interactions from a CSV file."""
    with session_scope() as session:
        report = ingest_interactions_csv(session, path)
        typer.echo(json.dumps(report.__dict__, indent=2))


@cli.command("ingest-artifacts")
def ingest_artifacts(manifest: Path) -> None:
    """Ingest artifacts from a manifest CSV file."""
    with session_scope() as session:
        report = ingest_artifacts_manifest(session, manifest)
        typer.echo(json.dumps(report.__dict__, indent=2))


@cli.command("extract-features")
def extract_features() -> None:
    """Extract features for all pending artifacts."""
    with session_scope() as session:
        count = extract_features_for_pending(session)
        typer.echo(json.dumps({"updated_artifacts": count}))


@cli.command("generate-synth")
def generate_synth(
    out_dir: Path = typer.Argument(..., help="Output directory for synthetic data"),
    n_entities: int = 500,
    n_events: int = 50_000,
    n_interactions: int = 20_000,
    n_artifacts: int = 2_000,
    seed: int = 42,
) -> None:
    """Generate a fully synthetic multimodal dataset."""
    generate_synthetic_dataset(
        out_dir=out_dir,
        n_entities=n_entities,
        n_events=n_events,
        n_interactions=n_interactions,
        n_artifacts=n_artifacts,
        seed=seed,
    )
    typer.echo(json.dumps({"status": "ok", "out_dir": str(out_dir)}))


@cli.command("train")
def train(config_path: Optional[Path] = None) -> None:
    """Train the multimodal model on the current database contents."""
    run_id, metrics = run_training(config_path=config_path)
    typer.echo(json.dumps({"run_id": run_id, "metrics": metrics}, indent=2))


@cli.command("reproduce")
def reproduce(run_id: str) -> None:
    """Reproduce a previous run and verify determinism."""
    report = reproduce_run(run_id)
    typer.echo(json.dumps(report, indent=2))


@cli.command("determinism-check")
def determinism_check() -> None:
    """Run an end-to-end determinism verification on a small synthetic subset."""
    report = run_determinism_check()
    typer.echo(json.dumps(report, indent=2))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
