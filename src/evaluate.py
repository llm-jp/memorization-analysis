import argparse
import logging
from pathlib import Path

import torch
import tqdm
from metrics import extractable
from transformers import AutoModelForCausalLM
from utils import PREFIX_LENGTHS, load_examples, save_examples

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


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load model from {args.model_name_or_path}")
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch_dtype)
    model.eval()
    logger.debug(model)

    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    logger.info(f"Create output directory {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.info(f"Load examples from {path}.")
        examples = [example for example in load_examples(path)]

        logger.info("Calculate memorization metrics for each step.")
        input_ids = torch.tensor([e.token_ids for e in examples])[..., :-1]
        for i in tqdm.trange(0, len(examples), args.batch_size):
            batch_examples = examples[i : i + args.batch_size]
            batch_input_ids = input_ids[i : i + args.batch_size]

            batch_input_ids = batch_input_ids.to(model.device)

            n = 50  # The number of tokens to complete.
            for l in PREFIX_LENGTHS:  # noqa: E741
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
                for example, extractable_ in zip(batch_examples, cur_extractable.tolist()):
                    example.metrics[f"extractable/{l}"] = extractable_

        logger.info("Save metrics.")
        output_file = output_dir / path.relative_to(data_dir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_examples(examples, output_file)
        logger.info(f"Saved metrics to {output_file}.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
