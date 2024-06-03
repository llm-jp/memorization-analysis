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

## Preprocess

First, preprocess the training data.

### Extract training data

The first step is to extract the training data at specific training steps.

```bash
PATH_TO_DATA_DIR=<PATH-TO-DATA-DIR>
PATH_TO_EXTRACT_DIR=<PATH-TO-EXTRACT-DIR>
python src/preprocess.py extract --data_dir $PATH_TO_DATA_DIR --output_dir $PATH_TO_EXTRACT_DIR
```

### Annotate training data

The second step is to annotate the training data with frequency information, etc.

```bash
PATH_TO_EXTRACT_DIR=<PATH-TO-EXTRACT-DIR>
PATH_TO_ANNOTATE_DIR=<PATH-TO-ANNOTATE-DIR>
python src/preprocess.py annotate --data_dir $PATH_TO_EXTRACT_DIR --output_dir $PATH_TO_ANNOTATE_DIR
```

## Evaluate memorization metrics

Evaluate memorization metrics for a model.

```bash
PATH_TO_ANNOTATE_DIR=<PATH-TO-ANNOTATE-DIR>
PATH_TO_RESULT_DIR=<PATH-TO-RESULT-DIR>
MODEL_NAME_OR_PATH=<MODEL-NAME-OR-PATH>
python src/evaluate.py --data_dir $PATH_TO_ANNOTATE_DIR --output_dir $PATH_TO_RESULT_DIR --model_name_or_path $MODEL_NAME_OR_PATH
```

- Merge the near-dup-count to the result.
```bash
 python3 src/merge-near-dup-pythia.py --threshold 0.8 --model_siz
e 12b
```

```bash
python src/merge_near_dup_count.py --data_dir $PATH_TO_RESULT_DIR
```


Visualize the results.

```bash
PATH_TO_RESULT_DIR=<PATH-TO-RESULT-DIR>
PATH_TO_PLOT_DIR=<PATH-TO-PLOT-DIR>
python src/plot.py --data_dir $PATH_TO_RESULT_DIR --output_dir $PATH_TO_PLOT_DIR
```

To browse the memorization metrics, run the following command and open the URL in a browser.

```bash
PATH_TO_RESULT_DIR=<PATH-TO-RESULT-DIR>
streamlit run src/browse.py -- --data_dir $PATH_TO_RESULT_DIR
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
