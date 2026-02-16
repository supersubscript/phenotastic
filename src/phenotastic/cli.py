"""Command-line interface for phenotastic.

This module provides CLI commands for running pipelines, managing configurations,
and visualizing meshes.
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option()
def cli() -> None:
    """Phenotastic: 3D plant phenotyping tools.

    Run pipelines, manage configurations, and visualize meshes from the command line.
    """


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="YAML pipeline configuration file",
)
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["standard", "high_quality", "mesh_only", "quick", "full"]),
    help="Use a preset pipeline instead of config file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--save-mesh/--no-save-mesh",
    default=True,
    help="Save output mesh to file",
)
@click.option(
    "--save-domains/--no-save-domains",
    default=True,
    help="Save domain data to CSV",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Print progress information",
)
def run(  # noqa: C901
    input_file: str,
    config: str | None,
    preset: str | None,
    output: str | None,
    save_mesh: bool,
    save_domains: bool,
    verbose: bool,
) -> None:
    """Run the phenotyping pipeline on an input file.

    INPUT_FILE can be a 3D image (TIFF) or mesh file (VTK, PLY, STL).

    Examples:

        # Run with a preset
        phenotastic run image.tif --preset standard --output results/

        # Run with custom config
        phenotastic run image.tif --config my_pipeline.yaml

        # Quick preview without saving
        phenotastic run image.tif --preset quick --no-save-mesh --no-save-domains
    """
    import numpy as np

    from phenotastic import PhenoMesh, Pipeline, load_preset

    # Determine which pipeline to use
    if config and preset:
        raise click.UsageError("Cannot specify both --config and --preset")

    if not config and not preset:
        preset = "standard"
        if verbose:
            click.echo("No config or preset specified, using 'standard' preset")

    # Load pipeline
    if preset:
        pipeline = load_preset(preset)
        if verbose:
            click.echo(f"Loaded preset: {preset}")
    else:
        pipeline = Pipeline.from_yaml(config)  # type: ignore[arg-type]
        if verbose:
            click.echo(f"Loaded config: {config}")

    if verbose:
        click.echo(f"Pipeline has {len(pipeline)} steps")

    # Determine input type and load
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()

    mesh_input: PhenoMesh | None = None
    image_input: np.ndarray | None = None

    if suffix in [".vtk", ".ply", ".stl", ".obj"]:
        # Load as mesh
        import pyvista as pv

        polydata = pv.read(str(input_path))
        mesh_input = PhenoMesh(polydata)
        if verbose:
            click.echo(f"Loaded mesh: {mesh_input.n_points} points, {mesh_input.n_faces} faces")
    else:
        # Load as image
        import tifffile as tiff

        image_input = tiff.imread(str(input_path))
        image_input = np.squeeze(image_input)
        if verbose:
            click.echo(f"Loaded image: shape {image_input.shape}, dtype {image_input.dtype}")

    # Run pipeline
    if verbose:
        click.echo("Running pipeline...")

    input_data: np.ndarray | PhenoMesh = mesh_input if mesh_input is not None else image_input  # type: ignore[assignment]
    result = pipeline.run(input_data, verbose=verbose)

    # Save outputs
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_mesh and result.mesh is not None:
            mesh_path = output_dir / f"{input_path.stem}_mesh.vtk"
            result.mesh.to_polydata().save(str(mesh_path))
            if verbose:
                click.echo(f"Saved mesh to: {mesh_path}")

        if save_domains and result.domain_data is not None:
            domains_path = output_dir / f"{input_path.stem}_domains.csv"
            result.domain_data.to_csv(str(domains_path), index=False)
            if verbose:
                click.echo(f"Saved domain data to: {domains_path}")

    # Print summary
    if verbose:
        click.echo("\n--- Results ---")
        if result.mesh is not None:
            click.echo(f"Mesh: {result.mesh.n_points} points, {result.mesh.n_faces} faces")
        if result.domains is not None:
            n_domains = len(np.unique(result.domains))
            click.echo(f"Domains: {n_domains} domains found")
        if result.domain_data is not None:
            click.echo(f"Domain data: {len(result.domain_data)} rows")

    click.echo("Done!")


@cli.command("init-config")
@click.argument("output_file", type=click.Path())
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["standard", "high_quality", "mesh_only", "quick", "full"]),
    default="standard",
    help="Preset to use as template",
)
def init_config(output_file: str, preset: str) -> None:
    """Generate a pipeline configuration file from a preset.

    Creates a YAML file that you can customize for your workflow.

    Examples:

        # Generate standard config
        phenotastic init-config my_pipeline.yaml

        # Generate high-quality config
        phenotastic init-config my_pipeline.yaml --preset high_quality
    """
    from phenotastic import get_preset_yaml

    yaml_content = get_preset_yaml(preset)

    output_path = Path(output_file)
    output_path.write_text(yaml_content)

    click.echo(f"Created configuration file: {output_path}")
    click.echo(f"Based on preset: {preset}")
    click.echo("\nEdit this file to customize your pipeline.")


@cli.command("list-operations")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["all", "contour", "mesh", "domain"]),
    default="all",
    help="Filter operations by category",
)
def list_operations(category: str) -> None:
    """List all available pipeline operations.

    Shows operation names that can be used in pipeline configuration files.

    Examples:

        # List all operations
        phenotastic list-operations

        # List only mesh operations
        phenotastic list-operations --category mesh
    """
    from phenotastic import OperationRegistry

    registry = OperationRegistry()
    operations = registry.list_operations(category)

    click.echo(f"\n{category.upper()} OPERATIONS ({len(operations)} total):\n")

    for op_name in operations:
        click.echo(f"  - {op_name}")

    click.echo("\nUse these names in your pipeline YAML configuration.")


@cli.command("list-presets")
def list_presets_cmd() -> None:
    """List available preset pipeline configurations."""
    from phenotastic import list_presets

    presets = list_presets()

    click.echo("\nAVAILABLE PRESETS:\n")

    descriptions = {
        "standard": "Balanced pipeline for typical workflows",
        "high_quality": "Multi-pass smoothing with aggressive filtering",
        "mesh_only": "For when you already have a mesh (skip contouring)",
        "quick": "Fast preview with minimal processing",
        "full": "Complete pipeline from image to domain analysis",
    }

    for preset in presets:
        desc = descriptions.get(preset, "")
        click.echo(f"  {preset:15} - {desc}")

    click.echo("\nUse with: phenotastic run image.tif --preset <name>")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str) -> None:
    """Validate a pipeline configuration file.

    Checks that the YAML is valid and all operations are recognized.

    Examples:

        phenotastic validate my_pipeline.yaml
    """
    from phenotastic import Pipeline
    from phenotastic.exceptions import ConfigurationError

    try:
        pipeline = Pipeline.from_yaml(config_file)
        warnings = pipeline.validate()

        click.echo(f"Valid configuration with {len(pipeline)} steps:\n")

        for i, step in enumerate(pipeline.steps, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in step.params.items())
            if params_str:
                click.echo(f"  {i:2}. {step.name}({params_str})")
            else:
                click.echo(f"  {i:2}. {step.name}")

        if warnings:
            click.echo("\nWarnings:")
            for warning in warnings:
                click.echo(f"  - {warning}")

        click.echo("\nConfiguration is valid!")

    except ConfigurationError as e:
        click.echo(f"Invalid configuration: {e}", err=True)
        raise SystemExit(1) from e


@cli.command()
@click.argument("mesh_file", type=click.Path(exists=True))
@click.option(
    "--scalars",
    "-s",
    help="Scalar array to color by (e.g., 'curvature', 'domains')",
)
@click.option(
    "--cmap",
    default="viridis",
    help="Colormap for scalar visualization",
)
def view(mesh_file: str, scalars: str | None, cmap: str) -> None:
    """Visualize a mesh file interactively.

    Opens an interactive 3D viewer for the mesh.

    Examples:

        # Basic view
        phenotastic view mesh.vtk

        # View with curvature coloring
        phenotastic view mesh.vtk --scalars curvature

        # View domains with custom colormap
        phenotastic view mesh.vtk --scalars domains --cmap tab20
    """
    import pyvista as pv

    from phenotastic import PhenoMesh

    polydata = pv.read(mesh_file)
    mesh = PhenoMesh(polydata)

    click.echo(f"Mesh: {mesh.n_points} points, {mesh.n_faces} faces")

    if scalars and scalars not in polydata.array_names:
        available = ", ".join(polydata.array_names) if polydata.array_names else "none"
        click.echo(f"Warning: '{scalars}' not found. Available: {available}")
        scalars = None

    mesh.plot(scalars=scalars, cmap=cmap)


def main() -> None:
    """Main entry point for the CLI."""
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, nl=False),
        format="{message}\n",
        level="INFO",
    )

    cli()


if __name__ == "__main__":
    main()
