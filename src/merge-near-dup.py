# insert /model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged's near_dup_count to /model/i-sugiura/memorization-analysis/llm-jp/result1.3B


import gzip
import json
from pathlib import Path

import click


@click.command()
@click.option("--threshold", default=0.6)
@click.option("--model_size", default="13B")
def main(threshold, model_size):
    DIR = f"/model/i-sugiura/memorization-analysis/llm-jp/near-dup/threshold_{threshold}/merged/"

    for dir_idx in range(10):
        for path_idx in range(12):
            merged_path = Path(f"{DIR}used_data_0{dir_idx}/used_data_{path_idx}.jsonl.gz")
            result_path = Path(
                f"/model/i-sugiura/memorization-analysis/llm-jp/result{model_size}/used_data_0{dir_idx}/used_data_{path_idx}.jsonl.gz"
            )
            # Load json line files
            with gzip.open(merged_path, "rt") as f:
                merged = [json.loads(line) for line in f]
            with gzip.open(result_path, "rt") as f:
                result = [json.loads(line) for line in f]
            for m, r in zip(merged, result):
                r["completion_stats"]["near_dup_count"] = m["completion_stats"]["near_dup_count"]
            # Save overwritten result

            with gzip.open(result_path, "wt") as f:
                for line in result:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
