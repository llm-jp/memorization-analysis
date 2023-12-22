from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import plotly.graph_objs as go
from utils import Example


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
    for step, examples in step_examples_map.items():
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
        for step, examples in step_examples_map.items():
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
            x=list(step_examples_map.keys()),
            y=list(reversed(list(key_l_map.values()))),
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Extractable fraction change over training steps",
        xaxis_title="Training steps",
        yaxis_title="Sequence length",
    )
    fig.write_image(str(path))
