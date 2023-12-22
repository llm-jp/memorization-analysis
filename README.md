# Memorization Analysis

This repository contains code to analyze the extent to which LLM-jp models memorize training data.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Run evaluation

Calculate memorization metrics for a model:

```bash
python src/evaluate.py \
    --data_dir <PATH-TO-DATA-DIR> \
    --output_dir <PATH-TO-OUTPUT-DIR> \
    [--verbose]
```

Use `--help` to see all available options.

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
