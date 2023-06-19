#!/bin/bash

#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --exclude=zeus[400,402]
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128000
#SBATCH -p GPU
#SBATCH --gres=gpu:K80:2
#SBATCH --time=05:00:00
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

script=$1
echo "Running...${script}"

srun python ${script} \
--epochs=10 \
--scheduler=True \
--lr0=1e-2 \
--lr_decay=0.5 \
--decay_freq=2 \
--batch_size=128 \
--print_freq=1000 \

kill %1
