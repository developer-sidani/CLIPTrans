#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --partition=global
#SBATCH --job-name=data_setup
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/job_name_%j.log
#SBATCH --error=/home/asidani/logs/job_name_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmad.sidani@studenti.polito.it

# Load the Python module
module load intel/python/3

# Function to send Discord notifications
function send_discord {
    python3 /home/asidani/message.py "$@"
}

# Source environment
source ~/.bashrc

# Starting the script
echo "[SCRIPT]: Starting the run"
send_discord "Starting the run for $SLURM_JOB_NAME"

# Change directory to target location
cd /home/asidani/CLIPTrans/data/multi30k

# # Clone and set up the dataset
# echo "[SCRIPT]: Setting up Multi30k dataset"
# send_discord "Setting up Multi30k dataset"

# send discord "Cloning the dataset"
# git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset
# mv multi30k-dataset/* .
# rm -rf multi30k-dataset
# mkdir images text
# mv data text
# mv scripts text
# cd text/data/task1/raw

# send_discord "Downloading the dataset"
# gunzip *.gz
# for file in val.*; do mv "$file" "test_2016_val.${file##*.}"; done
# cd ../image_splits
# mv val.txt test_2016_val.txt
# cd ../../../..

# send discord "Downloading the images"
# cp /home/asidani/thesis/data/flickr30k-images.tar images 
# cp /home/asidani/thesis/data/coco/test2017.zip images
# cp /home/asidani/thesis/data/mnt_images/test_2017-flickr-images.tar.gz images
cd images
# rm -rf flickr30k-images
# send_discord "Folder removed"
# unzip test2017.zip
# mv test2017 test_2017_mscoco
# tar -xvzf test_2017-flickr-images.tar.gz
# mv task1 test_2017_flickr
# tar -xf flickr30k-images.tar
# send_discord "Extracted the images!"
# mv flickr30k-images train
# mkdir test_2016_flickr test_2016_val
# mv ../create_test_val_flickr.py .

send_discord "Creating test and validation splits"
python3 create_test_val_flickr.py /home/asidani/CLIPTrans/data/multi30k/text/data/task1/image_splits/test_2016_val.txt /home/asidani/CLIPTrans/data/multi30k/images/train /home/asidani/CLIPTrans/data/multi30k/images/test_2016_val
python3 create_test_val_flickr.py /home/asidani/CLIPTrans/data/multi30k/text/data/task1/image_splits/test_2016_flickr.txt /home/asidani/CLIPTrans/data/multi30k/images/train /home/asidani/CLIPTrans/data/multi30k/images/test_2016_flickr


# Notify with log and error files
LOG_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/job_name_${SLURM_JOB_ID}.err"
python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"

# Script end
echo "[SCRIPT]: Run ended"
send_discord "Run ended successfully"