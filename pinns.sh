#!/bin/bash

#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --exclude=zeus[400,402]
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128000
#SBATCH -p GPU
#SBATCH --gres=gpu:K80:2
#SBATCH --time=16:00:00
#SBATCH -o ./slurm/slurm-%j.log

source ~/.torch.sh

PORT=$((RANDOM % 40001 + 10000))

export MASTER_PORT=$PORT

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
unset CUDA_VISIBLE_DEVICES

bash gpu_log.sh &

srun python pinns.py \
--epochs=25 \
--lr0=1e-3 \
--scheduler=True \
--decay_freq=3 \
--lr_decay=0.75 \
--m1='./models/flow5051121.pth' \
--m2='./models/phase5051120.pth' \
--batch_size=128 \
--print_freq=1000 \

kill %1
