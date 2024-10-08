import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator

from elastic_transport import ConnectionTimeout
from elasticsearch import Elasticsearch, helpers
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
    parent_parser.add_argument(
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

    parser_search = subparsers.add_parser("search", parents=[parent_parser])
    parser_search.add_argument(
        "--query",
        type=str,
        required=True,
        help="The token ids to search.",
    )
    parser_search.set_defaults(handler=search)

    parser_count = subparsers.add_parser("count", parents=[parent_parser])
    parser_count.add_argument(
        "--query",
        type=str,
        required=True,
        help="The token ids to count.",
    )
    parser_count.set_defaults(handler=count)

    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        exit(1)

    return args


def create_index(host: str, index: str) -> None:
    """Create an Elasticsearch index.

    Args:
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
    """
    es = Elasticsearch(host)

    if es.indices.exists(index=index):
        inp = input(f"Index {index} already exists. Do you want to delete it? [y/N] ")
        if inp.lower() == "y":
            es.options(request_timeout=2_400).indices.delete(index=index)
        else:
            logger.info("Aborting.")
            exit(0)

    es.options(request_timeout=2_400).indices.create(
        index=index,
        body={
            "settings": {
                "index": {"number_of_shards": 64, "number_of_replicas": 0},
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "tokenizer": "whitespace",
                        },
                    },
                },
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "iteration": {"type": "integer"},
                    "dataset_name": {"type": "keyword"},
                    "token_ids": {"type": "text", "analyzer": "custom_analyzer"},
                },
            },
        },
    )


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
            token_ids = " ".join(map(str, example.token_ids))
            yield {
                "_index": index,
                "_source": {
                    "iteration": example.iteration,
                    "dataset_name": example.dataset_name.split("/")[-1],
                    "token_ids": token_ids,
                },
            }

    while True:
        try:
            helpers.bulk(es.options(request_timeout=2_400), actions())
            break
        except ConnectionTimeout:
            logger.warning("Connection timeout. Retrying.")
            continue


def search_documents(host: str, index: str, body: dict, **kwargs) -> list[dict]:
    """Search for documents in an index.

    Args:
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        body (dict): The body of the request.
        **kwargs: Additional keyword arguments.

    Returns:
        list[dict]: The list of documents that match the query.
    """
    es = Elasticsearch(host)
    res = es.options(request_timeout=2_400).search(
        index=index,
        body=body,
        **kwargs,
    )
    return res["hits"]["hits"]


def count_documents(host: str, index: str, body: dict) -> int:
    """Count the number of documents in an index.

    Args:
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        body (dict): The body of the request.

    Returns:
        int: The number of documents in the index.
    """
    es = Elasticsearch(host)
    res = es.options(request_timeout=2_400).count(index=index, body=body)
    return res["count"]


def index(args: argparse.Namespace) -> None:
    """Index data to Elasticsearch.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    create_index(args.host, args.index)

    paths = []
    data_dir = Path(args.data_dir)
    for fold in FOLDS:
        for local_rank in LOCAL_RANKS:
            paths.append(data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz")

    worker_fn = partial(index_documents, args.host, args.index)

    with ProcessPoolExecutor(args.num_workers) as executor:
        for _ in executor.map(worker_fn, paths):
            pass


def search(args: argparse.Namespace) -> None:
    """Search for documents in an index.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    documents = search_documents(
        args.host,
        args.index,
        body={"query": {"match_phrase": {"token_ids": args.query}}},
        size=3,
    )
    for document in documents:
        print(document["_source"]["token_ids"])
        print("---")


def count(args: argparse.Namespace) -> None:
    """Count the number of documents in an index.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    num_documents = count_documents(
        args.host,
        args.index,
        body={"query": {"match_phrase": {"token_ids": args.query}}},
    )
    print(f"Found {num_documents} documents that match the query.")


def main(args: argparse.Namespace) -> None:
    args.handler(args)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
