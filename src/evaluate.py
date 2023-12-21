import argparse
import logging
from collections import defaultdict
from pathlib import Path

import tqdm
from utils import FOLDS, LOCAL_RANKS, load_examples

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logger.info(f"Data directory: {args.data_dir}")
    data_dir = Path(args.data_dir)

    for fold in FOLDS:
        step_examples_map = defaultdict(list)
        for local_rank in LOCAL_RANKS:
            data_file = (
                data_dir / f"used_data_{fold}" / f"used_data_{local_rank}.jsonl.gz"
            )
            logger.info(f"Load examples from {data_file}.")
            for example in tqdm.tqdm(load_examples(data_file)):
                if example.iteration % args.interval == 0:
                    step_examples_map[example.iteration].append(example)

            examples = next(iter(step_examples_map.values()))
            logger.info(f"Found {len(examples)} examples for each step.")
            if len(examples) >= args.num_examples_per_step:
                logger.info(
                    f"Found enough examples to sample {args.num_examples_per_step} "
                    "examples for each step, so skip loading the rest."
                )
                break
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing data files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1_000,
        help="The interval between two steps to sample examples.",
    )
    parser.add_argument(
        "--num_examples_per_step",
        type=int,
        default=128,
        help="The number of examples to sample for each step.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main(args)
