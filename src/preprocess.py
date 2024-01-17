import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from elastic_search import count_documents, search_documents
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
    for example in load_examples(path):
        if example.iteration % interval == 0:
            examples.append(example)
    return examples


def get_prefix_stats(
    example: Example,
    host: str,
    index: str,
) -> dict[int, dict[str, int]]:
    """Return prefix statistics.

    Args:
        example (Example): The example.
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.

    Returns:
        dict[int, dict[str, int]]: The prefix statistics.
    """
    prefix_stats = {}
    for prefix_length in PREFIX_LENGTHS:
        prefix = " ".join(map(str, example.token_ids[:prefix_length]))

        # Count the number of documents that contain the prefix.
        body = {"query": {"match_phrase": {"token_ids": prefix}}}
        count = count_documents(host, index, body=body)
        if count == 0:
            logger.warning(f"Prefix {prefix} not found in {index}.")

        # Get the last iteration of the prefix.
        if count == 0:
            last_iteration = -1
        elif count == 1:
            last_iteration = example.iteration
        else:
            body = {
                "query": {"match_phrase": {"token_ids": prefix}},
                "sort": [{"iteration": {"order": "desc"}}],
            }
            size = 1
            res = search_documents(
                host,
                index,
                body=body,
                size=size,
                max_concurrent_shard_requests=64,
            )
            if len(res) == 0:
                logger.warning(f"Prefix {prefix} not found in {index}.")
                last_iteration = -1
            else:
                last_iteration = res[0]["_source"]["iteration"]

        prefix_stats[prefix_length] = {
            "count": count,
            "last_iteration": last_iteration,
        }
    return prefix_stats


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
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
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

    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.info(f"Load examples from {path}.")
        examples = [example for example in load_examples(path)]

        logger.info("Get prefix statistics.")
        worker_fn = partial(get_prefix_stats, host=args.host, index=args.index)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for example, prefix_stats in zip(
                examples, executor.map(worker_fn, examples)
            ):
                example.prefix_stats = prefix_stats

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
