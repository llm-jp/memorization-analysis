import gzip
import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")

# path = "/model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged/used_data_00/used_data_0.jsonl.gz"
# path = "/model/i-sugiura/memorization-analysis/llm-jp/result13B/used_data_00/used_data_1.jsonl.gz"
model_size = "12b"

idx = 0
for i in range(142):
    start_idx = i * 1000
    end_idx = start_idx + 999
    path = f"/model/i-sugiura/memorization-analysis/pythia/result{model_size}/converted/pythia-{start_idx:05d}-{end_idx:05d}.jsonl.gz"

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            jsondata = json.loads(line)
            if (
                jsondata["completion_stats"]["count"] == 1
                and jsondata["completion_stats"]["near_dup_count"] == 1
                and jsondata["metrics"]["extractable/1000"] == True
            ):
                # return jsondata["text"].split(" ")
                tokens = jsondata["token_ids"][950:1000]
                json_obj = {}
                json_obj["tokne_ids[:950]"] = tokenizer.decode(jsondata["token_ids"][:950])
                json_obj["token_ids[950:1000]"] = tokenizer.decode(tokens)
                json_obj["count"] = jsondata["completion_stats"]["count"]
                json_obj["near-dup-count"] = jsondata["completion_stats"]["near_dup_count"]
                json_obj["extractable/1000"] = jsondata["metrics"]["extractable/1000"]
                json_obj["last_iteration"] = jsondata["completion_stats"]["last_iteration"]
                print(json_obj)


"""
tokenizer =  AutoTokenizer.from_pretrained("llm-jp/llm-jp-1.3b-v1.0")

token_ids = [0, 1, 2, 3, 3]
print(token_ids)
print(tokenizer.decode(token_ids))
"""
