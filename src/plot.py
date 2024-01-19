import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
from utils import PREFIX_LENGTHS, Example, load_examples

logger = logging.getLogger(__name__)

FREQUENCY_BINS = [0, 1, 10, 100, 1_000]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the output files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def plot_extractable(
    examples: list[Example],
    metric_key: str = "extractable",
    min_frequency: int = 0,
    max_frequency: int = 999_999_999_999,
) -> go.Figure:
    """Plot the extractable fraction of the examples.

    Args:
        examples (list[Example]): A list of examples.
        metric_key (str, optional): The metric key to plot. Defaults to "extractable".
        min_frequency (int, optional): The minimum frequency of the examples to plot.
        max_frequency (int, optional): The maximum frequency of the examples to plot.

    Returns:
        go.Figure: The plotly figure.
    """
    step_examples_map = defaultdict(list)
    for example in examples:
        step_examples_map[example.iteration].append(example)

    steps = sorted(step_examples_map.keys())

    z = []
    for l in PREFIX_LENGTHS:  # noqa: E741
        key = f"{metric_key}/{l}"
        row = []
        for step in steps:
            examples = []
            for example in step_examples_map[step]:
                if example.prefix_stats[str(l)]["last_iteration"] != step:
                    continue
                if example.prefix_stats[str(l)]["frequency"] < min_frequency:
                    continue
                if example.prefix_stats[str(l)]["frequency"] > max_frequency:
                    continue
                examples.append(example)
            if len(examples) == 0:
                row.append(np.nan)
                continue
            extractable = sum([e.metrics[key] for e in examples]) / len(examples)
            row.append(extractable)
        z.append(row)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(map(str, steps)),
            y=list(map(str, PREFIX_LENGTHS)),
        )
    )
    fig.update_layout(
        title="Extractable fraction change over training steps",
        xaxis_title="Training steps",
        yaxis_title="Sequence length",
    )
    return fig


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    examples = []
    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.info(f"Load examples from {path}.")
        for example in load_examples(path):
            examples.append(example)

    logger.info(f"Create output directory {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plot extractable.")
    path = output_dir / "extractable.png"
    fig = plot_extractable(examples)
    fig.write_image(path)
    logger.info(f"Saved to {path}.")
    for min_frequency, max_frequency in zip(FREQUENCY_BINS[:-1], FREQUENCY_BINS[1:]):
        path = output_dir / f"extractable_{min_frequency}_{max_frequency}.png"
        fig = plot_extractable(
            examples,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )
        fig.write_image(path)
        logger.info(f"Saved to {path}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
