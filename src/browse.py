import argparse
import logging
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import streamlit as st
from plot import plot_extractable
from streamlit_extras.stylable_container import stylable_container
from transformers import AutoTokenizer
from utils import Example, load_examples

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
        "--tokenizer_name_or_path",
        type=str,
        default="llm-jp/llm-jp-1.3b-v1.0",
        help="The name or path of the pretrained tokenizer.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print debug messages.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger.info(f"Load data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    examples = []
    for path in data_dir.glob("**/*.jsonl.gz"):
        logger.info(f"Load examples from {path}.")
        for example in load_examples(path):
            examples.append(example)

    logger.info(f"Create a tokenizer from '{args.tokenizer_name_or_path}'")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    step_seqlen_extractable_map: dict[tuple[int, int], list[Example]] = defaultdict(
        list
    )
    for example in examples:
        step = int(example.iteration)
        for metric_key in filter(
            lambda x: x.startswith("extractable"), example.metrics
        ):
            metric = example.metrics[metric_key]
            if metric is True:
                seqlen = int(metric_key.split("/")[1])
                step_seqlen_extractable_map[(step, seqlen)].append(example)
    step_seqlen_extractable_map = {
        key: value for key, value in sorted(step_seqlen_extractable_map.items())
    }

    st.title("Browse extractable examples")

    st.plotly_chart(plot_extractable(examples))

    choice = st.selectbox(
        "Select a grid (training step, sequence length)",
        list(step_seqlen_extractable_map.keys()),
    )

    if choice:
        step, seqlen = choice
        examples = step_seqlen_extractable_map[choice]
        st.header(f"Grid: {choice}")
        st.markdown(
            dedent(
                f"""\
                - Training step: {step:,}
                - Sequence length: {seqlen:,}
                - Number of extractable examples: {len(examples):,}
                """
            )
        )
        st.subheader("Extractable examples")
        for example in examples:
            prompt = tokenizer.decode(example.token_ids[: seqlen - 50])
            extracted = tokenizer.decode(example.token_ids[seqlen - 50 : seqlen])
            st.divider()
            st.markdown(f"**Source**: {example.dataset_name}")
            st.markdown(f"**Iteration**: {example.iteration}")
            st.markdown("**Prompt**")
            with stylable_container(
                "codeblock",
                "code {white-space: pre-wrap !important;",
            ):
                st.code(prompt)
            st.markdown("**Extracted**")
            with stylable_container(
                "codeblock",
                "code {white-space: pre-wrap !important;",
            ):
                st.code(extracted)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
