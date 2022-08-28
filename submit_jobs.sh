#!/bin/bash

#SBATCH --job-name=CNN
#SBATCH --output=/PHShome/yc703/Documents/postICH/logs.txt
#SBATCH --nodes=1
#SBATCH --partition=Medium
#SBATCH --qos=Medium
#SBATCH --chdir=/PHShome/yc703/Documents/postICH
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ychen146@mgh.harvard.edu
#SBATCH --array=[1-5]

user='/PHShome/yc703'
project='postICH'
experiment='structure'
# register 1-2
# img depths 1-2
# structure 1-5
# loss fun 1-2
# act_func 0-3
# learning_rate 1-2

module use /apps/modulefiles/conversion
module load python/3.8
source $user/.bashrc
conda activate postICH_gpu
cd $user/Documents/$project
export PYTHONPATH=$PYTHONPATH:$user/Documents/$project

chmod +x train/$experiment'.sh'
bash train/$experiment'.sh' $SLURM_ARRAY_TASK_ID
python utils/summarise_experiments.py --log_dir='logs' 
