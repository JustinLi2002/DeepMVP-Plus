#!/bin/bash
### Set the job name
#SBATCH --job-name=deepmvp_noppi
### Run in the partition named "general"
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
### To send email when the job is completed:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail
### Output and error logs
#SBATCH -e deepmvp_noppi.err
#SBATCH -o deepmvp_noppi.log
### Memory
#SBATCH --mem=100G
### Switch to the working directory; by default TORQUE launches processes
### from your home directory.

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

source /home/your-directory/my-venv/bin/activate

mkdir -p /your-directory/logs

echo "=== GPU INFO ==="
nvidia-smi
echo "================"

cd /your/work/directory
python deepmvp_reproduce_v2.py

echo "Done."
