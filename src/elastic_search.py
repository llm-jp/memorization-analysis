import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator

from elastic_transport import ConnectionTimeout
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from utils import FOLDS, LOCAL_RANKS, load_examples

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
        "--host",
        type=str,
        default="http://localhost:9200/",
        help="The Elasticsearch host.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="memorization-analysis-dev",
        help="The name of the Elasticsearch index.",
    )
    parent_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )

    parser_index = subparsers.add_parser("index", parents=[parent_parser])
    parser_index.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    parser_index.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers to use.",
    )
    parser_index.set_defaults(handler=index)

    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        exit(1)

    return args


def index_documents(host: str, index: str, path: Path) -> None:
    """Index documents to Elasticsearch.

    Args:
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        path (list[dict]): The list of documents to index.
    """
    es = Elasticsearch(host)

    def actions() -> Iterator[dict]:
        for example in load_examples(path):
            yield {
                "_index": index,
                "_source": {
                    "iteration": example.iteration,
                    "dataset_name": example.dataset_name.split("/")[-1],
                    "token_ids": example.token_ids,
                },
            }

    while True:
        try:
            helpers.bulk(es.options(request_timeout=1_200), actions())
            break
        except ConnectionTimeout:
            logger.warning("Connection timeout. Retrying.")
            continue


def index(args: argparse.Namespace) -> None:
    """Index data to Elasticsearch.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    es = Elasticsearch(args.host)

    if es.indices.exists(index=args.index):
        inp = input(
            f"Index {args.index} already exists. Do you want to delete it? [y/N] "
        )
        if inp.lower() == "y":
            es.indices.delete(index=args.index)
        else:
            logger.info("Aborting.")
            return

    es.indices.create(
        index=args.index,
        body={
            "settings": {"index": {"number_of_shards": 64, "number_of_replicas": 0}},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "iteration": {"type": "integer"},
                    "dataset_name": {"type": "keyword"},
                    "token_ids": {"type": "integer"},
                },
            },
        },
    )

    paths = []
    data_dir = Path(args.data_dir)
    for fold in FOLDS:
        for local_rank in LOCAL_RANKS:
            paths.append(
                data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
            )

    worker_fn = partial(index_documents, args.host, args.index)

    with ProcessPoolExecutor(args.num_workers) as executor:
        for _ in tqdm(executor.map(worker_fn, paths), total=len(paths)):
            pass


def main(args: argparse.Namespace) -> None:
    args.handler(args)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
