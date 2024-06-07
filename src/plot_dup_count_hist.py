# insert /model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged's near_dup_count to /model/i-sugiura/memorization-analysis/llm-jp/result1.3B


import gzip
import json
import os
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--threshold", default=0.8)
@click.option("--dataset", default="llm-jp")
def main(threshold, dataset):
    if dataset == "llm-jp":
        result_path = Path(f"/model/i-sugiura/memorization-analysis/llm-jp/near-dup/threshold_{threshold}/merged/")
    elif dataset == "pythia":
        result_path = Path(f"/model/i-sugiura/memorization-analysis/pythia/near-dup/threshold_{threshold}/merged/")
    # load every .jsonl.gz file in the directory and plot the histogram of  ["completion_stats"]["near_dup_count"]
    if dataset == "llm-jp":
        # used_data_*/*.jsonl
        files = list(result_path.glob("used_data_*/used_data_*.jsonl.gz"))
    else:
        files = list(result_path.glob("*.jsonl.gz"))
    near_count_list = []
    exact_count_list = []
    for file in files:
        with gzip.open(file, "rt") as f:
            result = [json.loads(line) for line in f]
        near_dup_counts = [r["completion_stats"]["near_dup_count"] for r in result]
        exact_count_list.extend([r["completion_stats"]["count"] for r in result])
        near_count_list.extend(near_dup_counts)
    # plot histogram
    print(len(near_count_list))
    print(len(exact_count_list))
    from collections import Counter

    import matplotlib.pyplot as plt

    count_dict = Counter(near_count_list)
    print("most_frequent near_dup_count", count_dict.most_common(10))
    print("most_frequent exact_dup_count", Counter(exact_count_list).most_common(10))
    plt.hist(near_count_list, bins=np.logspace(0, 5, 20), log=True, ec="black", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Near-duplication counts", fontsize=18)
    plt.ylabel("Number of examples", fontsize=18)
    plt.grid(linestyle="--")
    plt.tick_params(labelsize=18)
    plt.tight_layout()

    dir = f"{dataset}/{threshold}"
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, "near_dup_count_hist.png"))
    plt.show()
    plt.clf()

    plt.hist(exact_count_list, bins=np.logspace(0, 5, 20), log=True, ec="black", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Exact Duplication counts", fontsize=18)
    plt.ylabel("Number of examples", fontsize=18)
    plt.grid(linestyle="--")
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "exact_dup_count_hist.png"))
    plt.show()


if __name__ == "__main__":
    main()
