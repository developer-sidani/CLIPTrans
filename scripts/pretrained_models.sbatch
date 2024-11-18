#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=tst
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



source ~/.bashrc


# 2 Activate the virtual environment
conda activate cliptrans
# source activate /home/asidani/miniconda3/envs/cliptrans

# 3 Run the python script

echo "[SCRIPT]: Starting the run"

send_discord "[{SLURM_JOB_ID}]: Starting the run"

PRETRAINED_MODELS_INFERENCE_PARAMS=(
    --num_gpus 1
    --mn multi30k
    --src_lang en
    --tgt_lang fr
    --prefix_length 10
    --bs 32
    --test_ds 2016 flickr
    --stage translate
    --test
    --lm /home/asidani/thesis/pretrained_models/cliptrans/multi30k-en-fr/model_best_test_1.pth
)



cd /home/asidani/CLIPTrans/

send_discord "[{SLURM_JOB_ID}]: Executing the python script"
python3 src/main.py "${PRETRAINED_MODELS_INFERENCE_PARAMS[@]}"
send_discord "[{SLURM_JOB_ID}]: Pretrained models inference ended"





# Define the full path for log and error files
LOG_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.err"


python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"

echo "[SCRIPT]: Run ended"
