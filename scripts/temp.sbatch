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

# 2 Activate the virtual environment
source activate /home/asidani/.conda/envs/cliptrans

# 3 Run the python script

echo "[SCRIPT]: Temp!!! Starting the run"
send_discord "[${SLURM_JOB_ID}]: Temp!!! Starting the run"

# Define loop parameters
datasets=("2016 flickr")
languages=("de")

cd /home/asidani/CLIPTrans/

for dataset in "${datasets[@]}"; do
    # Split dataset into year and name
    read -r year ds_name <<< "$dataset"
    
    for lang in "${languages[@]}"; do
        # Run stages in order: caption then translate
        for stage in "translate"; do
            # Build parameters array
            params=(
                --num_gpus 1
                --mn multi30k
                --prefix_length 10
                --bs 32
                --update_count 4
                --lr 1e-5
                --test_ds "$year" "$ds_name"
                --stage translate
                --tgt_lang "$lang"
            )
            
            # Add model parameter for translation stage
            if [ "$stage" == "translate" ]; then
                params+=(--lm model_pretrained.pth)
            fi

            # Send start notification
            send_discord "[${SLURM_JOB_ID}]: Temp!!! Starting ${ds_name} ${year}, ${lang}, ${stage}"
            
            # Execute the command
            echo "[SCRIPT]: Temp!!! Running for ${ds_name} ${year}, ${lang}, ${stage}"
            /home/asidani/.conda/envs/cliptrans/bin/python3 src/main.py "${params[@]}"
            
            # Send completion notification
            send_discord "[${SLURM_JOB_ID}]: Temp!!! Completed ${ds_name} ${year}, ${lang}, ${stage}"
        done
    done
done
# Define the full path for log and error files
LOG_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.err"


python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"
