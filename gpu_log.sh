#!/bin/bash

# Set the desired logging interval in seconds
INTERVAL=30

# Set the log file path
LOG_FILE="./logs/nvidia-${SLURM_JOB_ID}.txt"

while true
do
    nvidia-smi >> $LOG_FILE
    sleep $INTERVAL
done
