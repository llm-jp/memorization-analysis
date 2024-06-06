# 変数の設定
model_name="llm-jp"
model_size="13B"
count_method="count" # "near_dup_count" or "count"
min_list=(1 2 10 100 1000)
max_list=(2 10 100 1000 10000)
# z_list=(0.03 0.02 0.4 0.8) # for pythia "count"
z_list=(0.2 0.2 0.01 0.1) # for llm-jp

# for
for i in {0..4}
do
    min_frequency=${min_list[$i]}
    max_frequency=${max_list[$i]}
    z_max=${z_list[$i]}
    data_dir="/model/i-sugiura/memorization-analysis/${model_name}/result${model_size}"
    output_dir="${model_name}/memorization/${model_size}/${count_method}/${min_frequency}-${max_frequency}"

    # Pythonスクリプトの実行
    python3.10 src/plot.py \
        --data_dir "$data_dir" \
        --output_dir "$output_dir" \
        --min_frequency "$min_frequency" \
        --max_frequency "$max_frequency" \
        --zmax "$z_max" \
        --count_method "$count_method"
done
