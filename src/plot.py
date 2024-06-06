import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
from utils import PREFIX_LENGTHS, Example, load_examples

logger = logging.getLogger(__name__)

FREQUENCY_BINS = [0, 1, 10, 100, 1_000, 10_000]

STEP_INTERVAL = 5_000


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
        "--least_num_examples_per_grid",
        type=int,
        default=10,
        help="The minimum number of examples to plot.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=0,
        help="The minimum frequency of the examples to plot.",
    )
    parser.add_argument(
        "--max_frequency",
        type=int,
        default=999_999_999_999,
        help="The maximum frequency of the examples to plot.",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=0,
        help="The maximum value of the heatmap.",
    )

    parser.add_argument(
        "--count_method",
        type=str,
        default="count",
        help="The method to count the number of examples.",
    )

    return parser.parse_args()


def plot_verbatim_memorization_ratio(
    examples: list[Example],
    metric_key: str = "extractable",
    min_frequency: int = 0,
    max_frequency: int = 999_999_999_999,
    least_num_examples_per_grid: int = 10,
    heatmap_zmax: float = 1,
    count_method: str = "count",  # or "near_dup_count"
) -> go.Figure:
    """Plot the verbatim memorization ratio.

    Args:
        examples (list[Example]): A list of examples.
        metric_key (str, optional): The metric key to plot. Defaults to "extractable".
        min_frequency (int, optional): The minimum frequency of the examples to plot.
        max_frequency (int, optional): The maximum frequency of the examples to plot.
        least_num_examples_per_grid (int, optional): The minimum number of examples to plot.

    Returns:
        go.Figure: The plotly figure.
    """

    step_examples_map = defaultdict(list)
    for example in examples:
        iteration = example.completion_stats["last_iteration"]
        if iteration < 0:
            continue
        iteration = (iteration // STEP_INTERVAL) * STEP_INTERVAL
        step_examples_map[iteration].append(example)

    steps = sorted(step_examples_map.keys())

    z = []
    for l in PREFIX_LENGTHS:  # noqa: E741
        key = f"{metric_key}/{l}"
        row = []
        for step in steps:
            examples = []
            for example in step_examples_map[step]:
                if min_frequency <= example.completion_stats[count_method] < max_frequency:
                    examples.append(example)
            if len(examples) < least_num_examples_per_grid:
                row.append(np.nan)
                continue
            memorization_ratio = sum([e.metrics[key] for e in examples]) / len(examples)
            row.append(memorization_ratio)
        z.append(row)
    # print(z)
    z_max = np.nanmax([np.nanmax(row) for row in z])
    logger.debug(f"z_max = {z_max:.3f}")
    z_min = np.nanmin([np.nanmin(row) for row in z])
    logger.debug(f"z_min = {z_min:.3f}")

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(map(str, steps)),
            y=list(map(str, PREFIX_LENGTHS)),
            zmin=0.0,
            zmax=np.nanmax((z_max, heatmap_zmax)),
        )
    )
    print(list(range(0, steps[-1], 10000)))
    fig.update_layout(
        xaxis_title="Training steps",
        yaxis_title="Sequence length",
        # 全ての文字サイズ 大, x軸は1000単位で表示
        xaxis = dict(
            tickfont = dict(size=20),
            tickvals = list(range(0, steps[-1]+1, 10000)),
        ),
        yaxis = dict(
            tickfont = dict(size=20),
        ),
        # heatmapの文字サイズ
        font = dict(size=20),
    )
    return fig


def plot_approximate_memorization_ratio(
    examples: list[Example],
    metric_key: str = "bleu",
    threshold: float = 0.75,
    min_frequency: int = 0,
    max_frequency: int = 999_999_999_999,
    least_num_examples_per_grid: int = 1,
    count_method: str = "count",  # or "near_dup_count"
) -> go.Figure:
    """Plot the approximate memorization ratio.

    Args:
        examples (list[Example]): A list of examples.
        metric_key (str, optional): The metric key to plot. Defaults to "bleu".
        threshold (float, optional): The threshold to consider as memorized. Defaults to 0.75.
        min_frequency (int, optional): The minimum frequency of the examples to plot.
        max_frequency (int, optional): The maximum frequency of the examples to plot.
        least_num_examples_per_grid (int, optional): The minimum number of examples to plot.

    Returns:
        go.Figure: The plotly figure.
    """
    step_examples_map = defaultdict(list)
    for example in examples:
        iteration = example.completion_stats["last_iteration"]
        if iteration < 0:
            continue
        iteration = (iteration // STEP_INTERVAL) * STEP_INTERVAL
        step_examples_map[iteration].append(example)

    steps = sorted(step_examples_map.keys())

    z = []
    for l in PREFIX_LENGTHS:  # noqa: E741
        key = f"{metric_key}/{l}"
        row = []
        for step in steps:
            examples = []
            for example in step_examples_map[step]:
                if min_frequency <= example.completion_stats[count_method] < max_frequency:
                    examples.append(example)
            if len(examples) < least_num_examples_per_grid:
                row.append(np.nan)
                continue
            memorization_ratio = sum([e.metrics[key] > threshold for e in examples]) / len(examples)
            row.append(memorization_ratio)
        z.append(row)

    z_max = np.nanmax([np.nanmax(row) for row in z])
    logger.debug(f"z_max = {z_max:.3f}")
    z_min = np.nanmin([np.nanmin(row) for row in z])
    logger.debug(f"z_min = {z_min:.3f}")

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(map(str, steps)),
            y=list(map(str, PREFIX_LENGTHS)),
            zmin=0.0,
        )
    )
    fig.update_layout(
        xaxis_title="Training steps",
        yaxis_title="Sequence length",
        # 全ての文字サイズ 大, x軸は1000単位で表示
        xaxis = dict(
            tickfont = dict(size=20),
            tickvals = list(range(0, steps[-1]+1, 10000)),
        ),
        yaxis = dict(
            tickfont = dict(size=20),
        ),
        # heatmapの文字サイズ
        font = dict(size=20),
    )
    return fig


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)
    examples = []
    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.debug(f"Load examples from {path}.")
        for example in load_examples(path):
            examples.append(example)

    logger.info(f"Create output directory {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #logger.info("Plot verbatim memorization ratio.")
    #path = output_dir / "verbatim_memorization_ratio.png"
    #fig = plot_verbatim_memorization_ratio(examples, count_method=args.count_method)
    #fig.write_image(path)
    #logger.info(f"Saved to {path}.")
    min_frequency = args.min_frequency
    max_frequency = args.max_frequency

    logger.info(f"Plot extractable with frequency in [{min_frequency}, {max_frequency}].")
    path = output_dir / f"verbatim_memorization_ratio.png"
    fig = plot_verbatim_memorization_ratio(
        examples,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        least_num_examples_per_grid=args.least_num_examples_per_grid,
        heatmap_zmax=args.zmax,
        count_method=args.count_method,
    )
    fig.write_image(path)
    logger.info(f"Saved to {path}.")

    #logger.info("Plot approximate memorization ratio.")
    #path = output_dir / "approximate_memorization_ratio.png"
    #fig = plot_approximate_memorization_ratio(examples)
    #fig.write_image(path)
    #logger.info(f"Saved to {path}.")
    #logger.info(f"Plot bleu with frequency in [{min_frequency}, {max_frequency}].")
    path = output_dir / f"approximate_memorization_ratio.png"
    fig = plot_approximate_memorization_ratio(
        examples,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        least_num_examples_per_grid=args.least_num_examples_per_grid,
        count_method=args.count_method,
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
