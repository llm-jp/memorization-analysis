start=$(date +%s)
export HF_HOME="/model/i-sugiura"
#PATH_TO_ANNOTATE_DIR="/model/kiyomaru/memorization-analysis/pythia/preprocess/annotate"
PATH_TO_ANNOTATE_DIR="/model/kiyomaru/memorization-analysis/preprocess/annotate/"
PATH_TO_RESULT_DIR="/model/i-sugiura/memorization-analysis/llm-jp/result1.3B"
MODEL_NAME_OR_PATH="llm-jp/llm-jp-1.3b-v1.0"
echo
python3.10 src/evaluate.py --data_dir $PATH_TO_ANNOTATE_DIR --output_dir $PATH_TO_RESULT_DIR --model_name_or_path $MODEL_NAME_OR_PATH --batch_size 12
end=$(date +%s)
runtime=$((end - start))
echo "Execution time: ${runtime} seconds"