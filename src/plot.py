import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import plotly.graph_objs as go
from utils import Example, load_examples

logger = logging.getLogger(__name__)


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


def plot_perplexity(
    examples: list[Example],
    path: Union[str, Path],
    metric_key: str = "perplexity",
) -> None:
    """Plot the perplexity of the examples.

    Args:
        examples (list[Example]): A list of examples.
        path (Union[str, Path]): Path to the output file.
        metric_key (str, optional): The metric key to plot. Defaults to "perplexity".
    """
    step_examples_map = defaultdict(list)
    for example in examples:
        assert metric_key in example.metrics
        step_examples_map[example.iteration].append(example)

    x = []
    y = []
    y_std = []
    for step, examples in sorted(step_examples_map.items()):
        x.append(step)
        perplexity = [example.metrics[metric_key] for example in examples]
        y.append(sum(perplexity) / len(perplexity))
        y_std.append(np.std(perplexity))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            error_y={
                "type": "data",
                "array": y_std,
                "visible": True,
            },
        )
    )
    fig.update_layout(
        title="Perplexity change over training steps",
        xaxis_title="Training steps",
        yaxis_title="Perplexity",
    )
    fig.write_image(str(path))


def plot_min_k_percent_prob(
    examples: list[Example],
    path: Union[str, Path],
    metric_key: str = "min_k_percent_prob",
) -> None:
    """Plot the Min-K% probability of the examples.

    Args:
        examples (list[Example]): A list of examples.
        path (Union[str, Path]): Path to the output file.
        metric_key (str, optional): The metric key to plot. Defaults to "min_k_percent_prob".
    """
    example = examples[0]
    key_k_map = {}
    for key in example.metrics:
        if key.startswith(metric_key):
            k = int(key.split("/")[1])
            key_k_map[key] = k

    step_examples_map = defaultdict(list)
    for example in examples:
        assert all(key in example.metrics for key in key_k_map)
        step_examples_map[example.iteration].append(example)

    fig = go.Figure()
    for key, k in key_k_map.items():
        x = []
        y = []
        y_std = []
        for step, examples in sorted(step_examples_map.items()):
            x.append(step)
            min_k_percent_prob = [example.metrics[key] for example in examples]
            y.append(sum(min_k_percent_prob) / len(min_k_percent_prob))
            y_std.append(np.std(min_k_percent_prob))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"K={k}",
                error_y={
                    "type": "data",
                    "array": y_std,
                    "visible": True,
                },
            )
        )
    fig.update_layout(
        title="Min-K% probability change over training steps",
        xaxis_title="Training steps",
        yaxis_title="Min-K% probability",
        showlegend=True,
    )
    fig.write_image(str(path))


def plot_extractable(
    examples: list[Example],
    path: Union[str, Path],
    metric_key: str = "extractable",
) -> None:
    """Plot the extractable fraction of the examples.

    Args:
        examples (list[Example]): A list of examples.
        path (Union[str, Path]): Path to the output file.
        metric_key (str, optional): The metric key to plot. Defaults to "extractable".
    """
    example = examples[0]
    key_l_map = {}
    for key in example.metrics:
        if key.startswith(metric_key):
            l = int(key.split("/")[1])  # noqa: E741
            key_l_map[key] = l

    step_examples_map = defaultdict(list)
    for example in examples:
        assert all(key in example.metrics for key in key_l_map)
        step_examples_map[example.iteration].append(example)

    z = []
    for key, l in sorted(key_l_map.items(), key=lambda x: x[1], reverse=True):
        row = []
        for step, examples in sorted(step_examples_map.items()):
            extractable_frac = sum(
                [example.metrics[key] for example in examples]
            ) / len(examples)
            row.append(extractable_frac)
        z.append(row)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[str(x_i) for x_i in step_examples_map.keys()],
            y=[str(y_i) for y_i in reversed(list(key_l_map.values()))],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Extractable fraction change over training steps",
        xaxis_title="Training steps",
        yaxis_title="Sequence length",
    )
    fig.write_image(str(path))


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

    logger.info("Plot perplexity.")
    path = output_dir / "perplexity.pdf"
    plot_perplexity(examples, path)
    logger.info(f"Saved to {path}.")

    logger.info("Plot min-k% probability.")
    path = output_dir / "min_k_percent_prob.pdf"
    plot_min_k_percent_prob(examples, path)
    logger.info(f"Saved to {path}.")

    logger.info("Plot extractable fraction.")
    path = output_dir / "extractable.pdf"
    plot_extractable(examples, path)
    logger.info(f"Saved to {path}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
