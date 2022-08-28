scan_num=$1
img_dir=$2
segment_script=$3
toolbox=$4

ID_list=$(cat $scan_num | awk -F, 'NR>1{print $2}')
log_file=$img_dir/$(basename $scan_num)
log_file=${log_file%'.csv'}
if [ -f $log_file ]
then
        rm $log_file
fi

mkdir -p $img_dir/'transform'
for ID in $ID_list
do
        ID=${ID%\"}
        ID=${ID#\"}
        echo 'segmenting '$ID >> $log_file
        one_out=$img_dir/$ID
        one_img=$one_out/$ID'.nii'

        # convert dicom into nii, rename
        python $segment_script/preprocess.py --dcm_paths=$scan_num \
                --save_dir=$img_dir --index=$ID
        matlab -nodesktop -nodisplay \
                -r "cd $segment_script; spm_segment('$toolbox','$one_img', '$one_out')"
        python $segment_script/postprocess.py --save_dir=$img_dir --index=$ID
        cp $one_out/y_*.nii $img_dir/'transform'/$ID'.nii'

        gzip_file=$img_dir/'transform'/$ID'.nii'
        if [ -f $gzip_file'.gz' ]
        then
                rm $gzip_file'.gz'
        fi
        gzip $gzip_file
        rm ~/java.log*
        rm $one_out -r
done
