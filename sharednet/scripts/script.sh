#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH -e results/logs/slurm-%j.err
#SBATCH -o results/logs/slurm-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com


eval "$(conda shell.bash hook)"

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/logs

##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh

# shellcheck disable=SC2046
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)"  --mode='train' --remark="practice" &

wait




