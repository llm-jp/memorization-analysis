import argparse
import logging
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils import COMPLETION_END_INDEX, COMPLETION_LENGTH, PREFIX_LENGTHS, Example, load_examples

from plot import FREQUENCY_BINS, STEP_INTERVAL, plot_extractable

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

    @st.cache_data
    def get_examples(data_dir: Path) -> list[Example]:
        examples = []
        for path in data_dir.glob("**/*.jsonl.gz"):
            logger.info(f"Load examples from {path}.")
            for example in load_examples(path):
                examples.append(example)
        return examples

    examples = get_examples(data_dir)

    logger.info(f"Create a tokenizer from '{args.tokenizer_name_or_path}'")

    @st.cache_resource
    def get_tokenizer(tokenizer_name_or_path: str) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    tokenizer = get_tokenizer(args.tokenizer_name_or_path)



    min_frequency, max_frequency = st.select_slider(
        "Select a frequency range",
        options=FREQUENCY_BINS,
        value=(0, 10_000),
    )
    step_seqlen_extractable_map: dict[tuple[int, int], list[Example]] = defaultdict(list)
    for example in examples:
        step = example.completion_stats["last_iteration"]
        if step < 0:
            continue
        if example.completion_stats["count"] < min_frequency:
            continue
        if example.completion_stats["count"] > max_frequency:
            continue
        step = (step // STEP_INTERVAL) * STEP_INTERVAL
        for prefix_length in PREFIX_LENGTHS:
            metric = example.metrics[f"extractable/{prefix_length}"]
            if metric is True:
                step_seqlen_extractable_map[(step, prefix_length)].append(example)
    step_seqlen_extractable_map = {key: value for key, value in sorted(step_seqlen_extractable_map.items())}

    st.title("Browse extractable examples")

    st.plotly_chart(
        plot_extractable(examples, min_frequency=min_frequency, max_frequency=max_frequency),
        theme=None,
    )

    step = st.selectbox(
        "Select a training step",
        sorted({key[0] for key in step_seqlen_extractable_map.keys()}),
    )

    seqlen = st.selectbox(
        "Select a sequence length",
        sorted({key[1] for key in step_seqlen_extractable_map.keys()}),
    )

    examples = step_seqlen_extractable_map.get((step, seqlen), [])
    st.header(f"Grid: ({step:,}, {seqlen:,})")
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
        if example.completion_stats["count"] < min_frequency:
            continue
        if example.completion_stats["count"] > max_frequency:
            continue

        start = COMPLETION_END_INDEX - seqlen
        end = COMPLETION_END_INDEX - COMPLETION_LENGTH
        prompt = tokenizer.decode(example.token_ids[start:end])

        start = COMPLETION_END_INDEX - COMPLETION_LENGTH
        end = COMPLETION_END_INDEX
        extracted = tokenizer.decode(example.token_ids[start:end])

        st.divider()

        st.markdown(f"**Source**: {example.dataset_name.split('/')[-1]}")
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

    if len(examples) == 0:
        st.warning("No extractable examples.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
