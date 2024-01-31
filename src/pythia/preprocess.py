import argparse
import gzip
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

from mmap_dataset import MMapIndexedDataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


START_ITERATION = 0
END_ITERATION = 143000
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 2049


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="The number of steps per file")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_path}")
    dataset = MMapIndexedDataset(args.data_path, skip_warmup=True)

    logger.info("Create tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m",
        revision="step3000",  # TODO: Do we need to specify this?
    )

    logger.info(f"Create output directory {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Extract examples")
    for start in trange(START_ITERATION, END_ITERATION, args.chunk_size):
        end = min(start + args.chunk_size, END_ITERATION)

        chunk_size = end - start

        # Load `chunk_size` batches at a time
        token_ids = dataset[start * BATCH_SIZE : end * BATCH_SIZE]

        # Reshape to (chunk_size, batch_size, sequence_length)
        token_ids = token_ids.reshape(chunk_size, BATCH_SIZE, SEQUENCE_LENGTH)

        # Decode token IDs in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            texts = list(
                tqdm(executor.map(tokenizer.batch_decode, token_ids), total=chunk_size, leave=False, desc="Decode")
            )

        # Write examples to file
        output_path = os.path.join(args.output_path, f"pythia-{start:05d}-{end - 1:05d}.jsonl.gz")
        logger.info(f"Write {len(token_ids) * BATCH_SIZE} examples to {output_path}")
        with gzip.open(output_path, "wt", encoding="utf-8") as output_file:
            for iteration, batch_token_ids, batch_texts in zip(
                trange(start, end, leave=False, desc="Write"), token_ids, texts
            ):
                for token_ids, text in zip(batch_token_ids, batch_texts):
                    example = {
                        "iteration": iteration,
                        "dataset_idx": 0,
                        "dataset_name": "pile",
                        "doc_ids": [0],
                        "text": text,
                        "token_ids": token_ids.tolist(),
                    }
                    output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
