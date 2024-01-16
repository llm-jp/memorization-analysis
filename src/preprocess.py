import argparse
import logging

# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
from pathlib import Path

import tqdm
from elastic_search import count_documents

# from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import PreTrainedTokenizer
from utils import (
    FOLDS,
    LOCAL_RANKS,
    PREFIX_LENGTHS,
    Example,
    load_examples,
    save_examples,
)

logger = logging.getLogger(__name__)


def get_prefix_frequencies(
    example: Example,
    host: str,
    index: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[int, int]:
    """Assign prefix frequencies to the example.

    Args:
        example (Example): The example.
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        dict[int, int]: The prefix frequencies.
    """
    prefix_frequency = {}
    for prefix_length in PREFIX_LENGTHS:
        prefix = tokenizer.decode(example.token_ids[:prefix_length])
        count = count_documents(host, index, prefix)
        prefix_frequency[prefix_length] = count
    return prefix_frequency


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
    parser_extract.set_defaults(handler=extract)
    # parser.add_argument(
    #     "--host",
    #     type=str,
    #     default="http://localhost:9200/",
    #     help="The Elasticsearch host.",
    # )
    # parser.add_argument(
    #     "--index",
    #     type=str,
    #     default="memorization-analysis-dev",
    #     help="The name of the Elasticsearch index.",
    # )
    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     default="llm-jp/llm-jp-1.3b-v1.0",
    #     help="The model name or path for the language model.",
    # )
    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=1,
    #     help="The number of workers to use.",
    # )
    return parser.parse_args()


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

    for fold in folds:
        for local_rank in LOCAL_RANKS:
            examples = []
            data_file = (
                data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
            )
            logger.info(f"Load examples from {data_file}.")
            for example in tqdm.tqdm(load_examples(data_file)):
                if example.iteration % args.interval == 0:
                    examples.append(example)

            logger.info("Save examples.")
            output_file = output_dir / data_file.relative_to(data_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            save_examples(examples, output_file)
            logger.info(f"Saved examples to {output_file}.")


def main(args: argparse.Namespace) -> None:
    args.handler(args)

    # logger.info("Create tokenizer.")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    #
    # logger.info(f"Load data from {args.data_dir}")
    # data_dir = Path(args.data_dir)
    #
    # logger.info(f"Create output directory {args.output_dir}")
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # if args.folds is None:
    #     folds = FOLDS
    # else:
    #     folds = args.folds
    #
    # for fold in folds:
    #     step_examples_map = defaultdict(list)
    #     for local_rank in LOCAL_RANKS:
    #         data_file = (
    #             data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
    #         )
    #         logger.info(f"Load examples from {data_file}.")
    #         for example in tqdm.tqdm(load_examples(data_file)):
    #             if example.iteration % args.interval == 0:
    #                 step_examples_map[example.iteration].append(example)
    #
    #         examples = next(iter(step_examples_map.values()))
    #         logger.info(f"Found {len(examples)} examples for each step.")
    #
    #     logger.info("Count frequencies of the prefix of each example.")
    #     worker_fn = partial(
    #         get_prefix_frequencies,
    #         host=args.host,
    #         index=args.index,
    #         tokenizer=tokenizer,
    #     )
    #     for examples in tqdm.tqdm(step_examples_map.values()):
    #         with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    #             for example, prefix_freqnencies in zip(
    #                 examples, executor.map(worker_fn, examples)
    #             ):
    #                 example.prefix_frequencies = prefix_freqnencies
    #
    #     logger.info("Save examples.")
    #     output_file = output_dir / f"examples_{fold}.jsonl.gz"
    #     examples = [
    #         example for examples in step_examples_map.values() for example in examples
    #     ]
    #     save_examples(examples, output_file)
    #     logger.info(f"Saved examples to {output_file}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
