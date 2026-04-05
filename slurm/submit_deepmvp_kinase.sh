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
#SBATCH --mail-user=justin.2.li@uconn.edu
#SBATCH -e deepmvp_ppi.err
#SBATCH -o deepmvp_ppi.log

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

source /home/FCAM/juli/HRP/my-venv/bin/activate

echo "=== GPU INFO ==="
nvidia-smi
echo "================"

cd /home/FCAM/juli/HRP
python deepmvpppi.py

echo "Done at" `date`
