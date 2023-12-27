import argparse
import logging

from elasticsearch import Elasticsearch

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
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )

    parser_index = subparsers.add_parser("index", parents=[parent_parser])
    parser_index.set_defaults(handler=index)

    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        exit(1)

    return args


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

    es.indices.create(index=args.index)


def main(args: argparse.Namespace) -> None:
    args.handler(args)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
