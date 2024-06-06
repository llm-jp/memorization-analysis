import gzip
import json

import click
from transformers import AutoTokenizer


@click.command()
@click.option("--sequence_length", default=100)
def main(sequence_length):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")

    # path = "/model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged/used_data_00/used_data_0.jsonl.gz"
    # path = "/model/i-sugiura/memorization-analysis/llm-jp/result13B/used_data_00/used_data_1.jsonl.gz"
    model_size = "12b"

    save_path = f"pythia/0.6/extractable_{sequence_length}.jsonl"

    idx = 0
    for i in range(142):
        start_idx = i * 1000
        end_idx = start_idx + 999
        path = f"/model/i-sugiura/memorization-analysis/pythia/result{model_size}/converted/pythia-{start_idx:05d}-{end_idx:05d}.jsonl.gz"
        with open(save_path, "a") as save_f:
            with gzip.open(path, "rt") as f:
                for i, line in enumerate(f):
                    jsondata = json.loads(line)
                    if (
                        jsondata["completion_stats"]["count"] == 1
                        and jsondata["completion_stats"]["near_dup_count"] == 1
                        and jsondata["metrics"][f"extractable/{sequence_length}"] == True
                    ):
                        # return jsondata["text"].split(" ")
                        tokens = jsondata["token_ids"][950:1000]
                        json_obj = {}
                        json_obj["tokne_ids[:950]"] = tokenizer.decode(jsondata["token_ids"][:950])
                        json_obj["token_ids[950:1000]"] = tokenizer.decode(tokens)
                        json_obj["count"] = jsondata["completion_stats"]["count"]
                        json_obj["near-dup-count"] = jsondata["completion_stats"]["near_dup_count"]
                        json_obj["extractable/{sequence_length}"] = jsondata["metrics"][
                            f"extractable/{sequence_length}"
                        ]
                        json_obj["last_iteration"] = jsondata["completion_stats"]["last_iteration"]
                        save_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
