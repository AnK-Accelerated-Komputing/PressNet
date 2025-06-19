#!bin/bash
# Training Data
OUTPUT_DIR="training_output"
TRAIN_DATA_PATH="/home/ubuntu/ujwal/PressNetTest/PressNet/surrogateAI/data/conical_press_dataset.h5"

# RUN Params
EPOCHS=2
CORE_MODEL="encode_process_decode"
BATCH_SIZE=1
LEARNING_RATE=1e-4
CUDA_INT=1

# MLFLOW Params
EXP_NAME="encode_decode_test"
IS_NESTED=False
# Activate the conda environment
conda activate graph_env
# Run the training script with the specified parameters
python3 surrogateAI/train.py \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --core_model $CORE_MODEL \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --cuda_int $CUDA_INT \
    --experiment_name $EXP_NAME \
    --train_data_path $TRAIN_DATA_PATH \
    --is_nested $IS_NESTED