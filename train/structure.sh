# Does image registration improves performance accuracy?
index=${1:-0}
prefix=''
extra='loss_type=regression'

variable_key='model_type'
variable_val=('densenet121', 'densenet169' 'resnet18' 'resnet34' 'CNN5', 'CNN6')

variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_'$variable_key'_'$index'/NCCT'

export PYTHONPATH=$PYTHONPATH:$pwd
python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
