threshold=0.8
model_size="12b"


#PATH_TO_RESULT_DIR="/model/i-sugiura/memorization-analysis/llm-jp/result1.3B"
#PATH_TO_PLOT_DIR="llm-jp/plot1.3B-near-dup-0.6"

PATH_TO_RESULT_DIR="/model/i-sugiura/memorization-analysis/pythia/result${model_size}"
PATH_TO_PLOT_DIR="pythia/plot${model_size}-near-dup-${threshold}"

# python3 src/plot.py --data_dir $PATH_TO_RESULT_DIR --output_dir $PATH_TO_PLOT_DIR
python3.10 src/plot.py --data_dir $PATH_TO_RESULT_DIR --output_dir $PATH_TO_PLOT_DIR --min_frequency 1 --max_frequency 10 --zmax 0.03 --count_method "near_dup_count"
