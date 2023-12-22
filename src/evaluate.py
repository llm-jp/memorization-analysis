import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import tqdm
from metrics import perplexity
from transformers import AutoModel, PreTrainedModel
from utils import FOLDS, LOCAL_RANKS, load_examples

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
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
        "--model_name_or_path",
        type=str,
        default="llm-jp/llm-jp-1.3b-v1.0",
        help="The model name or path for the language model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def logits(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Calculate logits for each token.

    Args:
        model (PreTrainedModel): The language model.
        input_ids (torch.Tensor): Input IDs of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Logits of shape (batch_size, sequence_length, vocab_size).
    """
    return model(input_ids).logits


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load model from {args.model_name_or_path}")
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.eval()
    logger.debug(model)

    logger.info(f"Load data from {args.data_dir}")
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

        logger.info(f"Sample {args.num_examples_per_step} examples for each step.")
        for step, examples in tqdm.tqdm(step_examples_map.items()):
            step_examples_map[step] = examples[: args.num_examples_per_step]

        logger.info("Calculate memorization metrics for each step.")
        metrics = []
        for step, examples in tqdm.tqdm(step_examples_map.items()):
            for i in range(1, len(examples), args.batch_size):
                batch_examples = examples[i : i + args.batch_size]
                batch_input_ids = torch.tensor(
                    [example.input_ids for example in batch_examples]
                )[..., :-1]
                batch_labels = torch.tensor(
                    [example.input_ids for example in batch_examples]
                )[..., 1:]
                batch_logits = logits(model, batch_input_ids)
                batch_perplexity = perplexity(batch_logits, batch_labels)
                metrics.extend(batch_perplexity.tolist())
        print(sum(metrics) / len(metrics), min(metrics), max(metrics))
        return


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
