#!/bin/bash

# Reserve 1 GPU
#SBATCH --gres=gpu:1

# Set maximum run time to two days
#SBATCH --time=3-00:00:00

# Run script on long partition (lowest priority)
#SBATCH --partition=long

# Reserve memory per node
#SBATCH --mem=6G

# Reserve one node
#SBATCH --nodes=1

# Reserve two CPU cores per task
#SBATCH --cpus-per-task=4

# Set name
#SBATCH --job-name=OVR_3

# Set notification email
#SBATCH --mail-user=s.reitsma@ru.nl
#SBATCH --mail-type=ALL

REMOTE_DIR=/vol/astro0/external_users/sreitsma
SCRATCH_DIR=/scratch/sreitsma/kaggle-diabetic-retinopathy
SCRIPT_DIR=/home/sreitsma/coma/kaggle-diabetic-retinopathy

# Create necessary directories
mkdir -p $SCRATCH_DIR
mkdir -p $SCRATCH_DIR/models

# Export necessary environmental variables
export LD_LIBRARY_PATH=/home/sreitsma/cudnn-6.5-linux-x64-v2:/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-6.5/bin:$PATH
export LIBRARY_PATH=/home/sreitsma/cudnn-6.5-linux-x64-v2:/usr/local/cuda-6.5/lib64:$LIBRARY_PATH
export CPATH=/home/sreitsma/cudnn-6.5-linux-x64-v2:/usr/local/cuda-6.5/lib64:$CPATH

# Post Slack status
curl --silent -X POST --data-urlencode "payload={\"channel\": \"#coma-status\", \"username\": \"coma-bot\", \"text\": \"Job '$SLURM_JOB_NAME' started on coma (job id: $SLURM_JOB_ID).\", \"icon_emoji\": \":rocket:\"}" https://hooks.slack.com/services/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

# Copy files from /vol/astro0 to /scratch over InfiniBand for local I/O access
echo "Copying files from /vol/astro0 to /scratch"
rsync -a --exclude="processed_512" --exclude="models" $REMOTE_DIR/* $SCRATCH_DIR/

# Start script and disable output buffering
echo "Running script"
stdbuf -oL -eL python $SCRIPT_DIR/deep/train.py $SLURM_JOB_NAME

# After script has finished
# Copy resulting model to /vol/astro0
echo "Copying model from /scratch to /vol/astro0"
cp -R $SCRATCH_DIR/models $REMOTE_DIR/

# Cleanup scratch
echo "Removing /scratch folders"
rm -r $SCRATCH_DIR/*

# Post Slack status
curl --silent -X POST --data-urlencode "payload={\"channel\": \"#coma-status\", \"username\": \"coma-bot\", \"text\": \"Job '$SLURM_JOB_NAME' finished on coma (job id: $SLURM_JOB_ID).\", \"icon_emoji\": \":rocket:\"}" https://hooks.slack.com/services/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
