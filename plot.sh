PATH_TO_RESULT_DIR="/model/i-sugiura/memorization-analysis/llm-jp/result1.3B"
PATH_TO_PLOT_DIR="llm-jp/plot1.3B-near-dup-0.6"
python3 src/plot.py --data_dir $PATH_TO_RESULT_DIR --output_dir $PATH_TO_PLOT_DIR
# python3 src/plot.py --data_dir $PATH_TO_RESULT_DIR --output_dir $PATH_TO_PLOT_DIR --min_frequency 1 --max_frequency 10 --zmax 0.03 --count_method "near_dup_count"