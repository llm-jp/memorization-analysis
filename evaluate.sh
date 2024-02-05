export HF_HOME="/model/i-sugiura"
PATH_TO_ANNOTATE_DIR="/model/kiyomaru/memorization-analysis/pythia/preprocess/annotate"
PATH_TO_RESULT_DIR="/model/i-sugiura/memorization-analysis/pythia/result6.9B"
MODEL_NAME_OR_PATH="EleutherAI/pythia-6.9b"
python3 src/evaluate.py --data_dir $PATH_TO_ANNOTATE_DIR --output_dir $PATH_TO_RESULT_DIR --model_name_or_path $MODEL_NAME_OR_PATH --batch_size 12
