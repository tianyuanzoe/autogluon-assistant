import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from hydra import compose, initialize
from omegaconf import OmegaConf
from rich import print as rprint
from typing_extensions import Annotated

from .assistant import TabularPredictionAssistant
from .task import TabularPredictionTask

__all__ = ["TabularPredictionAssistant", "TabularPredictionTask"]


def _resolve_config_path(path: str):
    print(Path.cwd())
    return os.path.relpath(Path(path), Path(__file__).parent.absolute())


def get_task(path: Path) -> TabularPredictionTask:
    """Get a task from a path."""

    return TabularPredictionTask.from_path(path)


def make_prediction_outputs(task: TabularPredictionTask, predictions: pd.DataFrame) -> pd.DataFrame:
    test_ids = task.test_data[task.test_id_column]
    output_ids = task.output_data[task.output_id_column]

    if not test_ids.equals(output_ids):
        rprint("[orange]Warning: Test IDs and output IDs do not match![/orange]")

    outputs = pd.concat(
        [
            task.test_data[task.test_id_column],
            predictions,
        ],
        axis="columns",
    )

    outputs.columns = task.output_columns
    return outputs


def run_assistant(
    config_path: Annotated[
        str, typer.Argument(help="Path to the configuration directory, which includes a config.yaml file")
    ],
    task_path: Annotated[str, typer.Argument(help="Directory where task files are included")],
    config_overrides: Annotated[Optional[str], typer.Option(help="Overrides for the config in Hydra format")] = "",
) -> str:
    """Run AutoGluon-Assistant on a task defined in a path."""
    rel_config_path = _resolve_config_path(config_path)
    with initialize(version_base=None, config_path=rel_config_path):
        config = compose(config_name="config", overrides=config_overrides.split(" "))

    rprint("🤖 [bold red] Welcome to AutoGluon-Assistant [/bold red]")

    rprint("Will use task config:")
    rprint(OmegaConf.to_container(config))

    task_path = Path(task_path).resolve()
    rprint(f"Task path: {task_path}")

    task = TabularPredictionTask.from_path(task_path)

    rprint("[green]Task loaded![/green]")
    rprint(task)

    assistant = TabularPredictionAssistant(config)
    task = assistant.preprocess_task(task)

    rprint("[green]Task preprocessing complete![/green]")

    assistant.fit_predictor(task)

    rprint("[green]Predictor fit complete![/green]")

    predictions = assistant.predict(task)

    output_filename = f"aga-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(output_filename, "w") as fp:
        make_prediction_outputs(task, predictions).to_csv(fp, index=False)

    rprint(f"[green]Prediction complete! Outputs written to {output_filename}[/green]")
    return output_filename


def main():
    typer.run(run_assistant)


if __name__ == "__main__":
    main()
