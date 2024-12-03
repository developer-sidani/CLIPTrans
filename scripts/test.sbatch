#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=cliptrans_infrerence
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/job_name_%j.log
#SBATCH --error=/home/asidani/logs/job_name_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmad.sidani@studenti.polito.it

###### 1 Load the module
module load nvidia/cudasdk/11.6
module load intel/python/3

function send_discord {
    python3 /home/asidani/message.py "$@"
}

echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"

# 2 Activate the virtual environment
source activate /home/asidani/.conda/envs/cliptrans


cd /home/asidani/CLIPTrans/
# 3 Define parameters for testing

DATASETS=("2016 flickr" "2017 flickr" "2018 flickr" "2017 mscoco")
LANGS=("de" "fr")
STAGES=("caption" "translate")

# Iterate through datasets, languages, and stages
for DS in "${DATASETS[@]}"; do
    for LANG in "${LANGS[@]}"; do
        for STAGE in "${STAGES[@]}"; do
            TEST_PARAMS=(
                --num_gpus 1
                --mn multi30k
                --src_lang en
                --tgt_lang $LANG
                --prefix_length 10
                --bs 32
                --test_ds "$DS"
                --stage $STAGE
                --test
                --lm model_pretrained.pth
            )

            echo "[SCRIPT]: Starting test for dataset $DS, target language $LANG, stage $STAGE"
            send_discord "[${SLURM_JOB_ID}]: Starting test for dataset $DS, target language $LANG, stage $STAGE"

            /home/asidani/.conda/envs/cliptrans/bin/python3 src/main.py "${TEST_PARAMS[@]}"

            send_discord "[${SLURM_JOB_ID}]: Finished test for dataset $DS, target language $LANG, stage $STAGE"
        done
    done
done

# Define the full path for log and error files
LOG_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"