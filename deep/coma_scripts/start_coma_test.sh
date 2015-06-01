#!/bin/bash

# Set maximum run time to two days
#SBATCH --time=00:01:00

# Run script on long partition (lowest priority)
#SBATCH --partition=short

# Reserve memory per node
#SBATCH --mem=1G

# Reserve one node
#SBATCH --nodes=1

# Reserve CPU cores per task
#SBATCH --cpus-per-task=1

# Set name
#SBATCH --job-name=TESTJOB

# Set notification email
#SBATCH --mail-user=s.reitsma@ru.nl
#SBATCH --mail-type=ALL

hostname

# Post Slack status
curl -X POST --data-urlencode "payload={\"channel\": \"#coma-status\", \"username\": \"coma-bot\", \"text\": \"Job '$SLURM_JOB_NAME' started on coma (job id: $SLURM_JOB_ID).\", \"icon_emoji\": \":rocket:\"}" https://hooks.slack.com/services/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

sleep 5