#!/bin/bash
#BSUB -J final_process
#BSUB -q normal
#BSUB -n 6
#BSUB -R rusage[mem=10000]

# After assigning the correct scan, run preprocessing the second time

# ====================input====================
user='/PHShome/yc703'
project='postICH'
raw_data='/data/rosanderson/is_ct/dicom/' # contain the images of all patients
# label data path:
metadata='data/original_labels/all.csv'
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

# ====================run scripts====================
# run through image preprocessing the second time
scan_num=$pro_data/original_labels/$(basename $raw_data)
scan_num=${scan_num%/}.csv
dcm_df=${scan_num%'.csv'}'_disamb.csv'
Rscript data_raw/remove_na.R $dcm_df
Rscript data_raw/split_data.R ${dcm_df%'.csv'}'2.csv' 12

dcm_df=${dcm_df%'.csv'}'2/sample0.csv'
img_dir=$(dirname $raw_data)/processed
segment_script=$user/Documents/$project/segmentation/
mkdir -p $img_dir
chmod +x $segment_script/segment_pipeline.sh 
bash $segment_script/segment_pipeline.sh $dcm_df $img_dir $segment_script $toolbox 

save_img_dir=$pro_data/'tmp_images/'
python $segment_script/trim_img.py --img_dir=$img_dir/'registered' \
        --save_dir=$save_img_dir/'reg_trim' --bindex='yes'
python $segment_script/trim_img.py --img_dir=$img_dir/'unregistered' \
        --save_dir=$save_img_dir/'unreg_trim' --shape='128,128'

python data_raw/find_shape.py --data_dir=$pro_data/'unreg_trim' \
         --save_dir=$pro_data/original_labels

# summarise dataframes into a single one, create train and test sets
Rscript data_raw/create_metadata.R $metadata \
        $pro_data/'original_labels/unreg_trim_shape.csv'
Rscript data_raw/train_test_split.R $pro_data/'original_labels/all_sel.csv' \
   $pro_data/'labels'
conda deactivate
