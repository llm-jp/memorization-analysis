import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import tqdm
from metrics import extractable, min_k_percent_prob, perplexity
from transformers import AutoModelForCausalLM
from utils import FOLDS, LOCAL_RANKS, load_examples, save_examples

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
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the output files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5_000,
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
        "--folds",
        nargs="+",
        type=str,
        required=False,
        help="The folds to evaluate. If not specified, all folds will be evaluated.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="The model name used in the output directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load model from {args.model_name_or_path}")
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch_dtype
    )
    model.eval()
    logger.debug(model)

    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    logger.info(f"Create output directory {args.output_dir}")
    if args.model_name is None:
        model_name = args.model_name_or_path.split("/")[-1]
    else:
        model_name = args.model_name
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.folds is None:
        folds = FOLDS
    else:
        folds = args.folds

    for fold in folds:
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
        for step, examples in tqdm.tqdm(step_examples_map.items()):
            input_ids = torch.tensor([e.token_ids for e in examples])[..., :-1]
            labels = torch.tensor([e.token_ids for e in examples])[..., 1:]
            for i in range(0, len(examples), args.batch_size):
                batch_examples = examples[i : i + args.batch_size]
                batch_input_ids = input_ids[i : i + args.batch_size]
                batch_labels = labels[i : i + args.batch_size]

                batch_input_ids = batch_input_ids.to(model.device)
                batch_labels = batch_labels.to(model.device)

                with torch.no_grad():
                    batch_logits = model(batch_input_ids).logits

                batch_perplexity = perplexity(batch_logits, batch_labels)
                for example, perplexity_ in zip(
                    batch_examples, batch_perplexity.tolist()
                ):
                    example.metrics["perplexity"] = perplexity_

                for k in [20]:  # Recommended in https://arxiv.org/abs/2310.16789.
                    batch_min_k_percent_prob = min_k_percent_prob(
                        batch_logits, batch_labels, k=k
                    )
                    for example, min_k_percent_prob_ in zip(
                        batch_examples, batch_min_k_percent_prob.tolist()
                    ):
                        example.metrics[f"min_k_percent_prob/{k}"] = min_k_percent_prob_

                n = 50  # The number of tokens to complete.
                for l in [100, 200, 500, 1_000]:  # noqa: E741
                    cur_input_ids = batch_input_ids[..., : l - n]
                    cur_labels = batch_input_ids[..., l - n : l]
                    with torch.no_grad():
                        cur_output_ids = model.generate(
                            cur_input_ids,
                            do_sample=False,
                            max_length=l,
                            eos_token_id=-100,  # Do not stop at EOS.
                            pad_token_id=-100,  # Do not stop at PAD.
                        )
                    cur_output_ids = cur_output_ids[..., l - n :]
                    cur_extractable = extractable(cur_output_ids, cur_labels)
                    for example, extractable_ in zip(
                        batch_examples, cur_extractable.tolist()
                    ):
                        example.metrics[f"extractable/{l}"] = extractable_

            for metric_key in examples[0].metrics:
                metrics = [e.metrics[metric_key] for e in examples]
                logger.info(
                    f"Step {step}: {metric_key} = {sum(metrics) / len(metrics):.4f}"
                )

        logger.info("Save metrics.")
        output_file = output_dir / f"metrics_{fold}.jsonl.gz"
        examples = [
            example for examples in step_examples_map.values() for example in examples
        ]
        save_examples(examples, output_file)
        logger.info(f"Saved metrics to {output_file}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
