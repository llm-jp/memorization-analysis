# Memorization Analysis

This repository contains code to analyze the extent to which LLM-jp models memorize training data.

## Requirements

- Python: 3.10+
- See [requirements.txt](requirements.txt) for Python package requirements.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Analyzing memorization metrics

First, calculate the memorization metrics for a model.

```bash
python src/evaluate.py --data_dir <PATH-TO-DATA-DIR> --output_dir <PATH-TO-OUTPUT-DIR>
```

Then, plot the memorization metrics.

```bash
python src/plot.py --data_dir <PATH-TO-DATA-DIR> --output_dir <PATH-TO-OUTPUT-DIR>
```

To browse the memorization metrics, run the following command and open the URL in a browser.

```bash
python src/browse.py --data_dir <PATH-TO-DATA-DIR>
```

## Development

Ensure that adding unit tests for new code and that all tests pass:

```bash
pytest -vv
```

The code is formatted using [Ruff](https://docs.astral.sh/ruff/).
To ensure that the code is formatted correctly, install the pre-commit hooks:

```bash
pre-commit install
```
