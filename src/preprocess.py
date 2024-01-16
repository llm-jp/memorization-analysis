import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import tqdm
from elastic_search import count_documents, search_documents
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils import (
    FOLDS,
    LOCAL_RANKS,
    PREFIX_LENGTHS,
    Example,
    load_examples,
    save_examples,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )

    parser_extract = subparsers.add_parser("extract", parents=[parent_parser])
    parser_extract.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    parser_extract.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the output files.",
    )
    parser_extract.add_argument(
        "--interval",
        type=int,
        default=5_000,
        help="The interval between two steps to sample examples.",
    )
    parser_extract.add_argument(
        "--folds",
        nargs="+",
        type=str,
        required=False,
        help="The folds to evaluate. If not specified, all folds will be evaluated.",
    )
    parser_extract.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers to use.",
    )
    parser_extract.set_defaults(handler=extract)

    parser_annotate = subparsers.add_parser("annotate", parents=[parent_parser])
    parser_annotate.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    parser_annotate.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the output files.",
    )
    parser_annotate.add_argument(
        "--host",
        type=str,
        default="http://localhost:9200/",
        help="The Elasticsearch host.",
    )
    parser_annotate.add_argument(
        "--index",
        type=str,
        default="memorization-analysis-dev",
        help="The name of the Elasticsearch index.",
    )
    parser_annotate.add_argument(
        "--model_name_or_path",
        type=str,
        default="llm-jp/llm-jp-1.3b-v1.0",
        help="The model name or path for the language model.",
    )
    parser_annotate.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers to use.",
    )
    parser_annotate.set_defaults(handler=annotate)

    return parser.parse_args()


def extract_examples(
    path: Path,
    interval: int,
) -> list[Example]:
    """Extract examples from a data file.

    Args:
        path (Path): The path of the data file.
        interval (int): The interval between two steps to sample examples.

    Returns:
        list[Example]: The extracted examples.
    """
    examples = []
    for example in tqdm.tqdm(load_examples(path)):
        if example.iteration % interval == 0:
            examples.append(example)
    return examples


def get_prefix_frequencies(
    example: Example,
    host: str,
    index: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[int, int]:
    """Return prefix frequencies to the example.

    Args:
        example (Example): The example.
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        dict[int, int]: The prefix frequencies.
    """
    prefix_frequencies = {}
    for prefix_length in PREFIX_LENGTHS:
        prefix = tokenizer.decode(example.token_ids[:prefix_length])
        count = count_documents(host, index, prefix)
        prefix_frequencies[prefix_length] = count
    return prefix_frequencies


def get_prefix_last_iterations(
    example: Example,
    host: str,
    index: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[int, int]:
    """Return the last iteration of each prefix.

    Args:
        example (Example): The example.
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        dict[int, int]: The last iteration of each prefix.
    """
    prefix_last_iterations = {}
    for prefix_length, count in example.prefix_frequencies.items():
        if count == 0:
            prefix_last_iterations[prefix_length] = None
        elif count == 1:
            prefix_last_iterations[prefix_length] = example.iteration
        else:
            prefix = tokenizer.decode(example.token_ids[:prefix_length])
            body = {
                "query": {"match_phrase": {"text": prefix}},
                "sort": [{"iteration": {"order": "desc"}}],
            }
            size = 1
            res = search_documents(host, index, body, size=size)
            prefix_last_iterations[prefix_length] = res[0]["_source"]["iteration"]
    return prefix_last_iterations


def extract(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    logger.info(f"Create output directory {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.folds is None:
        folds = FOLDS
    else:
        folds = args.folds

    data_files = []
    for fold in folds:
        for local_rank in LOCAL_RANKS:
            data_files.append(
                data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
            )

    logger.info("Extract examples.")
    worker_fn = partial(extract_examples, interval=args.interval)
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for data_file, examples in zip(
            data_files,
            executor.map(worker_fn, data_files),
        ):
            logger.info("Save examples.")
            output_file = output_dir / data_file.relative_to(data_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            save_examples(examples, output_file)
            logger.info(f"Saved examples to {output_file}.")


def annotate(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    logger.info(f"Create output directory {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Create tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.info(f"Load examples from {path}.")
        examples = [example for example in tqdm.tqdm(load_examples(path))]

        logger.info("Count frequencies of the prefix of each example.")
        worker_fn = partial(
            get_prefix_frequencies,
            host=args.host,
            index=args.index,
            tokenizer=tokenizer,
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for example, frequencies in zip(
                examples, executor.map(worker_fn, examples)
            ):
                example.prefix_frequencies = frequencies

        logger.info("Find the last iteration of each prefix.")
        worker_fn = partial(
            get_prefix_last_iterations,
            host=args.host,
            index=args.index,
            tokenizer=tokenizer,
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for example, iterations in zip(examples, executor.map(worker_fn, examples)):
                example.prefix_last_iterations = iterations

        logger.info("Save examples.")
        output_file = output_dir / path.relative_to(data_dir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_examples(examples, output_file)
        logger.info(f"Saved examples to {output_file}.")


def main(args: argparse.Namespace) -> None:
    args.handler(args)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
