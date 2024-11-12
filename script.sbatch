#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=tst
#SBATCH --mem=40GB
#SBATCH --output=logs/job_name_%j.log
#SBATCH --error=logs/job_name_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmad.sidani@studenti.polito.it

###### 1 Load the module
module load nvidia/cudasdk/11.6
module load intel/python/3

echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"

# 2 Activate the virtual environment
source activate /home/asidani/miniconda3/envs/env-ali

# 3 Run the python script

echo "[SCRIPT]: Starting the run"

TRAIN_PARAMS_YELP=(
    --style_a neg 
    --style_b pos
    --lang en
    --path_mono_A ./data/yelp/train.0.txt
    --path_mono_B ./data/yelp/train.1.txt
    --path_mono_A_eval ./data/yelp/dev.0.txt
    --path_mono_B_eval ./data/yelp/dev.1.txt
    --shuffle
    --generator_model_tag google-t5/t5-large
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_model ./classifiers/yelp/bert-base-uncased_5/
    --lambdas "10|1|1|1|1"
    --epochs 2
    --learning_rate 5e-5
    --max_sequence_length 64
    --batch_size 8 
    --save_base_folder ./ckpts/
    --save_steps 1
    --eval_strategy epochs
    --eval_steps 1
    --pin_memory
    --use_cuda_if_available
    --max_samples_train 100
    --max_samples_eval 50

)

TEST_PARAMS_YELP=(
    --style_a neg
    --style_b pos
    --lang en
    --path_paral_A_test ./data/yelp/test.0.txt
    --path_paral_B_test ./data/yelp/test.1.txt 
    --path_paral_test_ref ./data/yelp/references/test/
    --n_references 4
    --generator_model_tag google-t5/t5-large
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_eval ./classifiers/yelp/bert-base-uncased_5/
    --from_pretrained ./ckpts/epoch_1/
    --max_sequence_length 64
    --batch_size 16
    --pin_memory
    --use_cuda_if_available
    --max_samples_test 100

)

TRAIN_PARAMS_GYAFC=(
    --style_a informal
    --style_b formal
    --lang en
    --path_mono_A ./data/GYAFC/family_relationships/train.0.txt
    --path_mono_B ./data/GYAFC/family_relationships/train.1.txt
    --path_paral_A_eval ./data/GYAFC/family_relationships/dev.0.txt
    --path_paral_B_eval ./data/GYAFC/family_relationships/dev.1.txt
    --path_paral_eval_ref ./data/GYAFC/family_relationships/references/dev/
    --n_references 4
    --shuffle
    --generator_model_tag google-t5/t5-large
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_model ./classifiers/GYAFC/family_relationships/bert-base-cased_5/
    --lambdas "10|1|1|1|1"
    --epochs 2 
    --learning_rate 5e-5
    --max_sequence_length 64
    --batch_size 8
    --save_base_folder ./ckpts_gyafc/
    --save_steps 1
    --eval_strategy epochs
    --eval_steps 1
    --pin_memory
    --use_cuda_if_available
    --max_samples_train 100
    --max_samples_eval 50

)

TEST_PARAMS_GYAFC=(
    --style_a neg
    --style_b pos
    --lang en
    --path_paral_A_test ./data/yelp/test.0.txt
    --path_paral_B_test ./data/yelp/test.1.txt 
    --path_paral_test_ref ./data/yelp/references/test/
    --n_references 4
    --generator_model_tag google-t5/t5-large
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_eval ./classifiers/yelp/bert-base-uncased_5/
    --from_pretrained ./ckpts_gyafc/epoch_1/
    --max_sequence_length 64
    --batch_size 16
    --pin_memory
    --use_cuda_if_available
    --max_samples_test 100

)


TRAIN_PARAMS_SHAKESPEARE=(
    --style_a shakespearean
    --style_b modern
    --lang en
    --path_mono_A ./data/shakespeare/train.0.txt
    --path_mono_B ./data/shakespeare/train.1.txt
    --path_paral_A_eval ./data/shakespeare/dev.0.txt
    --path_paral_B_eval ./data/shakespeare/dev.1.txt
    --path_paral_eval_ref ./data/shakespeare/references/dev/
    --n_references 1
    --shuffle
    --generator_model_tag facebook/bart-base
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_model ./classifiers/shakespeare/bert-base-cased_10/
    --lambdas "1|1|1|1|1"
    --epochs 30 
    --learning_rate 5e-5
    --max_sequence_length 64
    --batch_size 64
    --save_base_folder ./ckpts_shakespear/
    --save_steps 1
    --eval_strategy epochs
    --eval_steps 1
    --pin_memory
    --use_cuda_if_available
    --comet_logging 1
    --comet_key "LR0zdShfxezpwALyFLRxdTys3"
    --comet_workspace "hadi-ibra"
    --comet_project_name "cycle_gan_shakespeare"

)

TEST_PARAMS_SHAKESPEARE=(
    --style_a neg
    --style_b pos
    --lang en
    --path_paral_A_test ./data/shakespeare/test.0.txt
    --path_paral_B_test ./data/shakespeare/test.1.txt 
    --path_paral_test_ref ./data/shakespeare/references/test/
    --n_references 4
    --generator_model_tag google-t5/t5-large
    --discriminator_model_tag distilbert-base-cased
    --pretrained_classifier_eval ./classifiers/yeshakespearelp/bert-base-uncased/
    --from_pretrained ./ckpts_gyafc/epoch_1/
    --max_sequence_length 64
    --batch_size 64
    --pin_memory
    --use_cuda_if_available
    --max_samples_test 100

)
TRAIN_PARAMS_CLASSIFIER=(
    --dataset_path "data/shakespeare/" 
    --max_sequence_length 200 
    --batch_size 32 
    --use_cuda_if_available 
    --learning_rate 5e-5 
    --epochs 10 
    --lr_scheduler_type "linear" 
    --model_tag "bert-base-cased" 
    --save_base_folder "/home/ihadi/TST-MoreStyles/" 
    --save_steps 1 
    --eval_strategy "epoch" 
    --eval_steps 1 
    --comet_logging 
    --comet_key "LR0zdShfxezpwALyFLRxdTys3" 
    --comet_workspace "hadi-ibra" 
    --comet_project_name "classification-model2"

)

# Train CLASIFIER
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 utils/train_classifier.py "${TRAIN_PARAMS_CLASSIFIER[@]}"

/home/asidani/miniconda3/envs/env-ali/bin/python3 main.py

# Train Yelp
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 train.py "${TRAIN_PARAMS_YELP[@]}"

# Test Yelp
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 test.py "${TEST_PARAMS_YELP[@]}"

# Train GYAFC
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 train.py "${TRAIN_PARAMS_GYAFC[@]}"

# Test GYAFC
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 test.py "${TEST_PARAMS_GYAFC[@]}"

# Train Shakespeare
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 train.py "${TRAIN_PARAMS_SHAKESPEARE[@]}"

# Test Shakespeare
# /home/ihadi/miniconda3/envs/cyclegan_tst/bin/python3 test.py "${TEST_PARAMS_SHAKESPEARE[@]}"

CURRENT_DIR=$(pwd)

# Define the full path for log and error files
LOG_FILE="${CURRENT_DIR}/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="${CURRENT_DIR}/logs/job_name_${SLURM_JOB_ID}.err"


/home/asidani/miniconda3/envs/env-ali/bin/python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"

echo "[SCRIPT]: Run ended"
