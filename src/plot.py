import argparse
import logging
from collections import defaultdict
from pathlib import Path

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


def plot_extractable(
    examples: list[Example],
    metric_key: str = "extractable",
) -> go.Figure:
    """Plot the extractable fraction of the examples.

    Args:
        examples (list[Example]): A list of examples.
        metric_key (str, optional): The metric key to plot. Defaults to "extractable".

    Returns:
        go.Figure: The plotly figure.
    """
    example = examples[0]
    key_l_map = {}
    for key in example.metrics:
        if key.startswith(metric_key):
            l = int(key.split("/")[1])  # noqa: E741
            key_l_map[key] = l
    key_l_map = {key: l for key, l in sorted(key_l_map.items(), key=lambda x: x[1])}

    step_examples_map = defaultdict(list)
    for example in examples:
        assert all(key in example.metrics for key in key_l_map)
        step_examples_map[example.iteration].append(example)
    step_examples_map = {
        step: examples for step, examples in sorted(step_examples_map.items())
    }

    z = []
    for key, l in key_l_map.items():
        row = []
        for step, examples in step_examples_map.items():
            extractable = sum([example.metrics[key] for example in examples]) / len(
                examples
            )
            row.append(extractable)
        z.append(row)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[str(x_i) for x_i in step_examples_map.keys()],
            y=[str(y_i) for y_i in key_l_map.values()],
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

    logger.info("Plot extractable fraction.")
    path = output_dir / "extractable.png"
    fig = plot_extractable(examples)
    fig.write_image(str(path))
    logger.info(f"Saved to {path}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
