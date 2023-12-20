import argparse
import datetime
import gzip
import random
import time

import pandas as pd
import simplejson as json
import torch
from evaluate import load
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1.3b")
parser.add_argument("--steps", type=int, default="1000")
args = parser.parse_args()

json_file = open("./settings/settings.json", "r")
json_data = json.load(json_file)

random.seed(42)
date = datetime.datetime.today()

start = time.time()  # 現在時刻（処理開始前）を取得


device = "cuda"

model_path = json_data["model_path"][args.model]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16
)

parent_path = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "refined"]
child_path = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

# Initialize
tmp_iteration_number = -1
ppl_tmp = []

use_iteration = 8787

perplexity = load("perplexity", module_type="metric")

f_id = 0

for parent_file_path in parent_path:
    parent_i = parent_path.index(parent_file_path)
    start_iteration = use_iteration * parent_i
    end_iteration = use_iteration * (parent_i + 1) - 1

    print(f"start_iteration = {start_iteration}, end_iteration = {end_iteration}")

    random_set = [random.randint(0, 11) for _ in range(128)]
    random_index = [[] for _ in range(12)]

    for i, r in enumerate(random_set):
        random_index[r].append(i)

    for child_file_path in child_path:
        filename = f"/model/llm-jp-search/data/used_data_20231102/used_data_{parent_file_path}/used_data_{child_file_path}.jsonl.gz"

        list_of_index = set(random_index[int(child_file_path)])

        collected_index = 0

        current_iteration = -1

        with gzip.open(filename, "r") as f:
            for line in f:
                obj = json.loads(line)

                # Extract training steps needed
                if obj["iteration"] % args.steps != 0:
                    continue
                else:
                    if obj["iteration"] != current_iteration:
                        print(obj["iteration"])
                        current_iteration = obj["iteration"]
                        collected_index = 0

                    if collected_index not in list_of_index:
                        collected_index += 1
                        continue
                    else:
                        collected_index += 1
                        if collected_index == 128:
                            collected_index = 0

                # Caluculate perplexity
                predictions = obj["text"]
                encodings = tokenizer(predictions, return_tensors="pt")
                max_length = model.config.n_positions
                stride = model.config.n_positions
                seq_len = encodings.input_ids.size(1)

                nlls = []
                prev_end_loc = 0
                for begin_loc in tqdm(range(0, seq_len, stride)):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = (
                        end_loc - prev_end_loc
                    )  # may be different from stride on last loop
                    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = model(input_ids, labels=target_ids)

                        # loss is calculated using CrossEntropyLoss which averages over valid labels
                        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                        # to the left by 1.
                        neg_log_likelihood = outputs.loss

                    nlls.append(neg_log_likelihood)

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

                ppl = torch.exp(torch.stack(nlls).mean())
                results = ppl.item()
                tmp = [obj["iteration"], obj["dataset_idx"], obj["doc_ids"], results]

                if tmp_iteration_number == -1:
                    tmp_iteration_number = obj["iteration"]
                    ppl_tmp.append(tmp)
                    # ppl_tmp.append([obj["iteration"], obj["doc_ids"], results["mean_perplexity"]])

                elif tmp_iteration_number == obj["iteration"]:
                    ppl_tmp.append(tmp)
                    # ppl_tmp.append([obj["iteration"], obj["doc_ids"].split(), results["mean_perplexity"]])
                else:
                    tmp_iteration_number = obj["iteration"]
                    ppl_tmp.append(tmp)
                    # ppl_tmp.append([obj["iteration"], obj["doc_ids"].split(), results["mean_perplexity"]])

    # 1セット後にcsv書き込み
    print("write csv")
    print(
        f"{parent_file_path}, {child_file_path}, start_iteration = {start_iteration}, end_iteration = {end_iteration}"
    )
    tmp_iteration_number = -1
    df = pd.DataFrame(
        ppl_tmp,
        index=None,
        columns=["iteration", "dataset_idx", "doc_ids", "perplexity"],
    )
    df["doc_ids"] = df["doc_ids"].apply(lambda x: "\n".join(map(str, x)))
    df.sort_values(by="iteration", inplace=True)
    if args.model == "1.3b":
        file_name_model = "1_3b"
    elif args.model == "13b":
        file_name_model = "13b"
    df.to_csv(
        f"./result/{file_name_model}/{date}.csv",
        mode="a",
        index=False,
        header=False,
        float_format="%.3f",
        encoding="utf-8_sig",
    )
    ppl_tmp = []


end = time.time()  # 現在時刻（処理完了後）を取得

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)
print(date)
