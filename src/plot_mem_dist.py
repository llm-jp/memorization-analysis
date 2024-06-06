import click
import matplotlib.pyplot as plt

from pathlib import Path
import gzip
import json
from collections import defaultdict
import numpy as np

@click.command()
@click.option("--model_name", default="pythia")
@click.option("--model_size", default="12b")

def main(model_name, model_size):
    data_dir = f"/model/i-sugiura/memorization-analysis/{model_name}/result{model_size}"
    data_dir = Path(data_dir)
    # (exact_duplication, near_duplication, 100_memorize, 200_memorize, 500_memorize, 1000_memorize,blue_100
    examples = []
    path_list = data_dir.glob("**/*.jsonl.gz")
    path_list = sorted(path_list)
    # only ten files
    #path_list = path_list[:50]
    for path in path_list:
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                example = defaultdict(dict)
                example["exact_duplication"] = data["completion_stats"]["count"]
                example["near_duplication"] = data["completion_stats"]["near_dup_count"]
                for prefix_length in [100, 200, 500, 1000]:
                    example["verbatim"][prefix_length] = data["metrics"][f"extractable/{prefix_length}"]
                    if model_name == "pythia":
                        example["approximate"][prefix_length] = data["metrics"][f"bleu/{prefix_length}"]
                examples.append(example)

    # plot exact duplication 1, 2-9, 10-100, 100-1000, 1000-10000 memorization ratio
    range_list = [1, 2, 10, 100, 1000, 10000]


    count_methods = ["exact_duplication", "near_duplication"]

    for count_method in count_methods:

        for prefix_length in [100, 200, 500, 1000]:
            ratio_list = [[] for _ in range(len(range_list) - 1)]
            for example in examples:
                for start, end in zip(range_list[:-1], range_list[1:]):
                    if start <= example[count_method] < end:
                        idx = range_list.index(end) - 1
                        ratio_list[idx].append(example["verbatim"][prefix_length])
                        break


            verbatim_mem_ratio_list = [sum(ratio) / len(ratio) for ratio in ratio_list]
            plt.plot(np.arange(1, 6), verbatim_mem_ratio_list, label=f"{prefix_length}")
        if count_method == "exact_duplication":
            plt.xlabel("Exact duplication", fontsize=16)
        else:
            plt.xlabel("Near duplication", fontsize=16)
        plt.ylabel("Verbatim memorization ratio", fontsize=16)

        plt.xticks([1, 2, 3, 4, 5], ["1", "2-9", "$10^1-10^2$", "$10^2-10^3$", "$10^3-10^4$"])
        plt.legend(title="Prefix length", fontsize=14, title_fontsize=16)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(f"{model_name}/memorization/{model_size}/{count_method}_verbatim_ratio.png")
        plt.show()
        plt.clf()

        if model_name == "pythia":
            for prefix_length in [100, 200, 500, 1000]:
                ratio_list = [[] for _ in range(len(range_list) - 1)]
                for example in examples:
                    for start, end in zip(range_list[:-1], range_list[1:]):
                        if start <= example[count_method] < end:
                            idx = range_list.index(end) - 1
                            ratio_list[idx].append(example["approximate"][prefix_length])
                            break


                mem_ratio_list = [sum(ratio) / len(ratio) for ratio in ratio_list]
                plt.plot(np.arange(1, 6), mem_ratio_list, label=f"{prefix_length}")
            if count_method == "exact_duplication":
                plt.xlabel("Exact duplication", fontsize=16)
            else:
                plt.xlabel("Near duplication", fontsize=16)
            plt.ylabel("Approximate memorization ratio", fontsize=16)

            plt.xticks([1, 2, 3, 4, 5], ["1", "2-9", "$10^1-10^2$", "$10^2-10^3$", "$10^3-10^4$"])
            plt.legend(title="Prefix length", fontsize=14, title_fontsize=16)
            plt.tick_params(labelsize=14)
            plt.tight_layout()
            plt.savefig(f"{model_name}/memorization/{model_size}/{count_method}_approximate_ratio.png")
            plt.show()
            plt.clf()


        plt.bar(np.arange(1, 6), [len(ratio) for ratio in ratio_list])
        if count_method == "exact_duplication":
            plt.xlabel("Exact duplication", fontsize=16)
        else:
            plt.xlabel("Near duplication", fontsize=16)
        plt.ylabel("Number of examples", fontsize=16)
        plt.xticks([1, 2, 3, 4, 5], ["1", "2-9", "$10^1-10^2$", "$10^2-10^3$", "$10^3-10^4$"])
        plt.yscale("log")
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(f"{model_name}/memorization/{model_size}/{count_method}_num.png")
        plt.show()
        plt.clf()



if __name__ == "__main__":
    main()
