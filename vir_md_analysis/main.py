import typer
from pathlib import Path
from vir_md_analysis.features.feature_extraction import extract_features

app = typer.Typer()


@app.command()
def main(
    traj: Path = typer.Argument(..., help="Path to trajectory file (.nc, .dcd, etc.)"),
    top: Path = typer.Argument(..., help="Path to topology file (.prmtop, .pdb, etc.)"),
    system: str = typer.Option(None, help="System name for output"),
    start: int = typer.Option(0, help="Start frame"),
    end: int = typer.Option(None, help="End frame"),
    outdir: Path = typer.Option(".", help="Output directory"),
    outfile: str = typer.Option(None, help="Output CSV filename"),
):
    """Extract MD features from a trajectory."""
    path, result = extract_features(trajectory_file=traj,
                                    topology_file=top,
                                    system_name=system,
                                    start_frame=start,
                                    end_frame=end,
                                    output_dir=outdir,
                                    save_fname=outfile,
                                    save_results=True,
                                    return_path=True
                                    )
    typer.echo(f"Feature extraction complete. Output: {path}")


if __name__ == "__main__":
    app()
