import argparse
import logging
from pathlib import Path

from utils import FOLDS, LOCAL_RANKS, load_examples

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    args = parser.parse_args()

    logger.info(f"Data directory: {args.data_dir}")
    data_dir = Path(args.data_dir)

    for fold in FOLDS:
        for local_rank in LOCAL_RANKS:
            data_file = (
                data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
            )
            logger.info(f"Load examples from {data_file}.")
            step_examples_map = load_examples(data_file)
            logger.debug(f"Loaded examples of {len(step_examples_map)} steps.")
            return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
