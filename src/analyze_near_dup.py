import gzip
import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-1.3b-v1.0")

# path = "/model/i-sugiura/memorization-analysis/llm-jp/near-dup/merged/used_data_00/used_data_0.jsonl.gz"
# path = "/model/i-sugiura/memorization-analysis/llm-jp/result13B/used_data_00/used_data_1.jsonl.gz"
idx = 0
for dir_idx in range(10):
    for file_idx in range(12):
        path = f"/model/i-sugiura/memorization-analysis/llm-jp/result13B/used_data_{dir_idx:02d}/used_data_{file_idx}.jsonl.gz"
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
