seg_script=$1
toolbox=$2

# create a binary mask for each labelled structure
template1=$toolbox/'template/Atlas_MNI152/ArterialAtlas.nii'
template2=$toolbox/'template/Atlas_MNI152/ArterialAtlas_level2.nii'
python $seg_script/label2img.py --img_path=$template1 \
        --save_path=$toolbox'/template/level1'
python $seg_script/label2img.py --img_path=$template2 \
        --save_path=$toolbox'/template/level2'
# generate the transformation from MNI space to the template space in CTseg
matlab -nodesktop -nodisplay \
        -r "cd $seg_script; MNItrans('$toolbox','$template1')"

# warp the binary mask to the template used by CTseg
template=$toolbox/'template/level1'
dir_out=$toolbox/'template/level1_warp'
mkdir -p $dir_out
matlab -nodesktop -nodisplay \
        -r "cd $seg_script; warp_atlas('$toolbox','$template','$dir_out')"
# combine the binary masks together
python $seg_script/img2label.py --img_dir=$dir_out \
        --save_path=$toolbox'/template/level1.nii'
rm $dir_out -r

template=$toolbox/'template/level2'
dir_out=$toolbox/'template/level2_warp'
mkdir -p $dir_out
matlab -nodesktop -nodisplay \
        -r "cd $seg_script; warp_atlas('$toolbox','$template','$dir_out')"
python $seg_script/img2label.py --img_dir=$dir_out \
        --save_path=$toolbox'/template/level2.nii'
rm $dir_out -r
