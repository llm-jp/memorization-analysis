import argparse
import gzip
import json
import os

from mmap_dataset import MMapIndexedDataset
from tqdm import trange
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    description="",
)
parser.add_argument("--start_iteration", type=int, default=0, help="What train step to start logging")
parser.add_argument("--end_iteration", type=int, default=143000, help="Train step to end logging (inclusive)")
parser.add_argument(
    "--load_path",
    type=str,
    default="/mnt/ssd-1/pile_preshuffled/standard/document",
    help=("MMap dataset path with .bin and .idx files. Omit the .bin (or) .idx " "Extension while specifying the path"),
)
parser.add_argument("--save_path", type=str, default="token_indicies", help="Save path for files")
parser.add_argument("--base_iteration", type=int, default=0, help="Base iteration")


# 143 files


def convert_to_llm_jp_format(pythia_data_path, llm_jp_output_dir, base_iteration=0):
    # Create output directory if not exists
    os.makedirs(llm_jp_output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m",
        revision="step3000",
        cache_dir="/model/i-sugiura/pythia-14m/step3000",
    )
    dataset = MMapIndexedDataset(pythia_data_path, skip_warmup=True)
    print(len(dataset))
    steps_per_file = 1000
    for i in trange(base_iteration, 143):
        path = os.path.join(llm_jp_output_dir, f"pythia-{i*steps_per_file:05d}-{(i+1)*steps_per_file-1:05d}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as output_file:
            for j in trange(steps_per_file):
                iteration = i * steps_per_file + j
                current_file_lines = []
                # TODO: end is (iteration+1)*1024 + 1 ? or not?
                batch = dataset[iteration * 1024 : (iteration + 1) * 1024]
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


# Example usage
args = parser.parse_args()

pythia_data_path = "/model/i-sugiura/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/merged/document"
llm_jp_output_dir = "/model/i-sugiura/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/converted"

convert_to_llm_jp_format(pythia_data_path, llm_jp_output_dir, base_iteration=args.base_iteration)
