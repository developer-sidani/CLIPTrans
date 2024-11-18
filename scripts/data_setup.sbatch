#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --partition=global
#SBATCH --job-name=configuration
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/job_name_%j.log
#SBATCH --error=/home/asidani/logs/job_name_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmad.sidani@studenti.polito.it

###### 1 Load the module
module load intel/python/3

function send_discord {
    python3 /home/asidani/discord_notify.py "$@"
}


source ~/.bashrc

# 3 Run the python script

echo "[SCRIPT]: Starting the run"

send_discord "Starting the run"


cd /home/asidani/CLIPTrans/data/multi30k


./setup_multi30k.sh


send_discord "Installation Ended"

CURRENT_DIR=$(pwd)

# Define the full path for log and error files
LOG_FILE="${CURRENT_DIR}/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="${CURRENT_DIR}/logs/job_name_${SLURM_JOB_ID}.err"


python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"

echo "[SCRIPT]: Run ended"
