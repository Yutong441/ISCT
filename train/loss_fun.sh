index=${1:-1}
prefix='dense'
prefix='save_prefix=results/'$prefix'_loss_'$index'/NCCT'

if [ $index == 0 ]; then
    extra='loss_type=regression;outcome_col=LSW;pos_label=4.5'
elif [ $index == 1 ]; then
    extra='loss_type=regression;outcome_col=LSW_log;pos_label=1.5'
elif [ $index == 2 ]; then
    extra='loss_type=classification;outcome_col=LSW_bool;pos_label=0.5;predict_class=2' 
fi

python3 utils/one_exp.py --config=$prefix';'$extra
