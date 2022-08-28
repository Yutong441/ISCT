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
# summarise all available scans of each patient in a csv file
echo 'summarising scans'
raw_nii=$pro_data/raw_nii
### python data_raw/CTname.py --scan_dir=$raw_data \
###         --log_dir=$pro_data/original_labels/ \
###         --save_dir=$raw_nii

# assign one scan to each patient at a given time point uniquely, 
# using regular expression
echo 'assigning scans to patients'
scan_num=$pro_data/original_labels/$(basename $raw_data)
scan_num=${scan_num%/}.csv
###Rscript data_raw/CTname.R $scan_num $metadata $raw_data

dcm_df=${scan_num%'.csv'}'_path.csv'
###Rscript data_raw/split_data.R $dcm_df 20

# convert dicom into nifty, select slices in 5-6mm interval
# then skull stripping, windowing, downsizing, normalising
echo 'segmentation'
segment_script=$user/Documents/$project/segmentation/
img_dir=$(dirname $raw_data)/processed

dcm_df=${dcm_df%'.csv'}'/sample0.csv'
mkdir -p $img_dir
#chmod +x $segment_script/segment_pipeline.sh 
#bash $segment_script/segment_pipeline.sh $dcm_df $img_dir $segment_script $toolbox 

### echo 'trim image'
### save_img_dir=$pro_data/'tmp_images/'
### mkdir -p $save_img_dir
### python $segment_script/trim_img.py --img_dir=$img_dir/'registered' \
###         --save_dir=$save_img_dir/'reg_trim' --bindex='yes'
### python $segment_script/trim_img.py --img_dir=$img_dir/'unregistered' \
###         --save_dir=$save_img_dir/'unreg_trim' --shape='128,128'
### 
### # obtain the shape of each image
### python data_raw/find_shape.py --data_dir=$img_dir/'reg_trim' \
###         --save_dir=$pro_data/original_labels
### python data_raw/find_shape.py --data_dir=$img_dir/'unreg_trim' \
###         --save_dir=$pro_data/original_labels 
 
# append those images with depths < 15 to the disambiguation list
### Rscript data_raw/missing_study.R $scan_num $raw_data
### conda deactivate
