#!/bin/bash

#SBATCH --job-name=setup
#SBATCH --output=/PHShome/yc703/Documents/postICH/logs/setup.txt
#SBATCH --partition=Short
#SBATCH --qos=Short
#SBATCH --chdir=/PHShome/yc703/Documents/postICH
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ychen146@mgh.harvard.edu

module use /apps/modulefiles/conversion
module load anaconda/4.11.0
conda init bash
source /PHShome/yc703/.bashrc
conda create -n postICH_gpu python=3.8
conda activate postICH_gpu
conda install numpy pandas matplotlib -c conda-forge
conda install pydicom nibabel plotnine medcam ipywidgets -c conda-forge
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch
conda install scikit-learn -c anaconda
conda deactivate
