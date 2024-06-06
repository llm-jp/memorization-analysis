# insert /model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged's near_dup_count to /model/i-sugiura/memorization-analysis/llm-jp/result1.3B


import gzip
import json
import os
from pathlib import Path

import click

threshold = 0.8
model_size = "12b"


@click.command()
@click.option("--threshold", default=0.6)
@click.option("--model_size", default="12b")
def main(threshold, model_size):
    DIR = f"/model/i-sugiura/memorization-analysis/pythia/near-dup/threshold_{threshold}/merged/"
    for i in range(142):
        start_idx = i * 1000
        end_idx = start_idx + 999
        merged_path = os.path.join(DIR, f"pythia-{start_idx:05d}-{end_idx:05d}.jsonl.gz")
        result_path = Path(
            f"/model/i-sugiura/memorization-analysis/pythia/result{model_size}/converted/pythia-{start_idx:05d}-{end_idx:05d}.jsonl.gz"
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
