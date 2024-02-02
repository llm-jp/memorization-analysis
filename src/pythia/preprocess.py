import json
import gzip
import os
import numpy as np
from transformers import AutoTokenizer


from utils.mmap_dataset import MMapIndexedDataset
from tqdm import trange
import numpy as np
import argparse
import os

import concurrent.futures

import logging
logger = logging.getLogger(__name__)




parser = argparse.ArgumentParser(
    description="",
)

parser.add_argument(
    "--load_path",
    type = str,
    default = '/mnt/ssd-1/pile_preshuffled/standard/document',
    help = ("MMap dataset path with .bin and .idx files. Omit the .bin (or) .idx "
            "Extension while specifying the path")
)
parser.add_argument(
    "--save_path",
    type=str,
    default="token_indicies",
    help="Save path for files"
)
parser.add_argument(
    "--base_file",
    type=int,
    default=0,
    help="Base file"
)

parser.add_argument(
    "--end_file",
    type=int,
    default=143,
    help="End file"
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print debug level logs"
)

parser.add_argument(
    "--num_processes",
    type=int,
    default=48,
    help="Number of processes"
)


def process(file_idx):
    pythia_data_path = "/model/i-sugiura/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/merged/document"
    llm_jp_output_dir = "/model/i-sugiura/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/converted"
    convert_to_llm_jp_format(pythia_data_path, llm_jp_output_dir, file_idx)
    logger.info(f"Finished processing file {file_idx}")

# 143 files

def convert_to_llm_jp_format(pythia_data_path, llm_jp_output_dir, file_idx):
    # Create output directory if not exists
    os.makedirs(llm_jp_output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m",
        revision="step3000",
        cache_dir="/model/i-sugiura/pythia-14m/step3000",
    )
    dataset = MMapIndexedDataset(pythia_data_path, skip_warmup = True)
    print(len(dataset))
    steps_per_file = 1000
    i = file_idx
    path = os.path.join(llm_jp_output_dir, f"pythia-{i*steps_per_file:05d}-{(i+1)*steps_per_file-1:05d}.jsonl.gz")
    with gzip.open(path, 'wt', encoding='utf-8') as output_file:
        for j in trange(steps_per_file):
            iteration = i * steps_per_file + j
            # TODO: end is (iteration+1)*1024 + 1 ? or not?
            batch = dataset[iteration*1024: (iteration+1)*1024]
            for data in batch:
                text = tokenizer.decode(data)
                token_ids = data.tolist()
                formatted_data = {
                    "iteration": iteration,
                    "dataset_idx": 0,
                    "dataset_name": "pile",
                    "doc_ids": [0],
                    "text": text,
                    "token_ids": token_ids
                }
                output_file.write(json.dumps(formatted_data, ensure_ascii=False))
                # output_file.write("\n")
                output_file.write("\n")


if __name__ == "__main__":

    args = parser.parse_args()

    max_files = args.end_file
    min_files = args.base_file
    max_processes = args.num_processes
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    with concurrent.futures.ProcessPoolExecutor(max_processes) as executor:
        for i in range(min_files, max_files):
            executor.submit(process, i)

