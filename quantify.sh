#!/bin/bash
#BSUB -J initial_process
#BSUB -q normal
#BSUB -n 6
#BSUB -R rusage[mem=10000]

# ====================input====================
user='/PHShome/yc703'
project='postICH'
raw_data='/data/rosanderson/is_ct/dicom/' # contain the images of all patients
# label data path:
metadata='/data/rosanderson/is_ct/phenotype_files/curated_cohort_data/metadata_all_assigned_clean.csv'
skullstrip_path=$user/'Documents/StripSkullCT' #script for skullstripping
pro_data=$user/'Documents/'$project'/data/' # output directory path
toolbox=$user/Documents

# ====================activate environment====================
module load python/3.8
module load R/4.1.0
module load MATLAB/2021b

source $user/.bashrc
conda activate postICH
cd $user/Documents/$project
export PYTHONPATH=$PYTHONPATH:$user/Documents/$project

# ==================================================
# warp arterial territory template into CTseg template space
if [ ! -f $user/'Documents/template/level1.nii' ]
then
        seg_script=$user/Documents/$project/make_atlas
        chmod +x $seg_script/warp_atlas.sh
        bash $seg_script/warp_atlas.sh $seg_script $toolbox
fi

# segment
img_dir=$(dirname $raw_data)/processed
python segmentation/seg_terr.py --img_dir=$img_dir/'registered' \
        --mask_dir=$img_dir/'registered_maps' \
        --template_dir=$toolbox/'template' \
        --save_dir=$pro_data/'quant'
