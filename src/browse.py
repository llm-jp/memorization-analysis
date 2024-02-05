import argparse
import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st
from plot import FREQUENCY_BINS, STEP_INTERVAL, plot_approximate_memorization_ratio
from streamlit_extras.stylable_container import stylable_container
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils import COMPLETION_END_INDEX, COMPLETION_LENGTH, PREFIX_LENGTHS, Example, load_examples

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

    st.title("Browse memorized examples")

    memorization_threshold = st.slider(
        "Select a memorization threshold",
        min_value=0.75,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    min_frequency, max_frequency = st.select_slider(
        "Select a frequency range",
        options=FREQUENCY_BINS,
        value=(0, 10_000),
    )
    memorized_examples: dict[tuple[int, int], list[Example]] = defaultdict(list)

    for example in examples:
        count = example.completion_stats["count"]
        if count < min_frequency or count > max_frequency:
            continue

        step = example.completion_stats["last_iteration"]
        if step < 0:
            continue
        step = (step // STEP_INTERVAL) * STEP_INTERVAL

        for prefix_length in PREFIX_LENGTHS:
            if memorization_threshold == 1.0:
                is_memorized = example.metrics[f"extractable/{prefix_length}"]
            else:
                is_memorized = example.metrics[f"bleu/{prefix_length}"] >= memorization_threshold
            if is_memorized:
                memorized_examples[(step, prefix_length)].append(example)
    memorized_examples = {key: value for key, value in sorted(memorized_examples.items())}

    st.plotly_chart(
        plot_approximate_memorization_ratio(
            examples,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            threshold=memorization_threshold,
        ),
        theme=None,
    )

    st.subheader("Memorized examples")

    step = st.selectbox(
        "Select a training step",
        sorted({key[0] for key in memorized_examples.keys()}),
    )

    seqlen = st.selectbox(
        "Select a sequence length",
        sorted({key[1] for key in memorized_examples.keys()}),
    )

    examples = memorized_examples.get((step, seqlen), [])
    st.markdown(f"**Number of memorized examples**: {len(examples):,}")

    for example in examples:
        start = COMPLETION_END_INDEX - seqlen
        end = COMPLETION_END_INDEX - COMPLETION_LENGTH
        prefix = tokenizer.decode(example.token_ids[start:end])

        start = COMPLETION_END_INDEX - COMPLETION_LENGTH
        end = COMPLETION_END_INDEX
        suffix = tokenizer.decode(example.token_ids[start:end])

        count = example.completion_stats["count"]

        completion = tokenizer.decode(example.completions[str(seqlen)])  # noqa

        bleu = example.metrics[f"bleu/{seqlen}"]

        st.divider()

        st.markdown(f"**Source**: {example.dataset_name.split('/')[-1]}")
        st.markdown("**Prefix**")
        with stylable_container(
            "codeblock",
            "code {white-space: pre-wrap !important;",
        ):
            st.code(prefix)
        st.markdown(f"**Suffix** (Count in corpus: {count:,})")
        with stylable_container(
            "codeblock",
            "code {white-space: pre-wrap !important;",
        ):
            st.code(suffix)
        st.markdown(f"**Completion** (BLEU: {bleu:.3f})")
        with stylable_container(
            "codeblock",
            "code {white-space: pre-wrap !important;",
        ):
            st.code(completion)

    if len(examples) == 0:
        st.warning("No memorized examples exist.")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
