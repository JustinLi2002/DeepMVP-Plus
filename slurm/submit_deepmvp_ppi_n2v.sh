#!/bin/bash
#SBATCH --job-name=deepmvp_ppi
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -e deepmvp_ppi.err
#SBATCH -o deepmvp_ppi.log

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

source /home/FCAM/your-driectory/my-venv/bin/activate

echo "=== GPU INFO ==="
nvidia-smi
echo "================"

cd /your/work/directory
python deepmvp_ppi.py

echo "Done at" `date`
