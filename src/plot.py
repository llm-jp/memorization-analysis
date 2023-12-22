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
