import argparse
import gzip
import json
import logging
import os

from mmap_dataset import MMapIndexedDataset
from tqdm import trange
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


START_ITERATION = 0
END_ITERATION = 143000
BATCH_SIZE = 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Create output directory if not exists
    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m",
        revision="step3000",
        cache_dir="/model/i-sugiura/pythia-14m/step3000",
    )
    dataset = MMapIndexedDataset(args.data_path, skip_warmup=True)
    logger.info(len(dataset))
    steps_per_file = 1000
    for i in trange(0, 143):
        path = os.path.join(args.output_path, f"pythia-{i*steps_per_file:05d}-{(i+1)*steps_per_file-1:05d}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as output_file:
            for j in trange(steps_per_file):
                iteration = i * steps_per_file + j
                current_file_lines = []
                # TODO: end is (iteration+1)*1024 + 1 ? or not?
                batch = dataset[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE]
                for data in batch:
                    text = tokenizer.decode(data)
                    token_ids = data.tolist()
                    formatted_data = {
                        "iteration": iteration,
                        "dataset_idx": 0,
                        "dataset_name": "pile",
                        "doc_ids": [0],
                        "text": text,
                        "token_ids": token_ids,
                    }
                    current_file_lines.append(json.dumps(formatted_data, ensure_ascii=False))
                output_file.write("\n".join(current_file_lines))


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
