# impact of learning rate on model performance
index=${1:-1}
prefix='dense'
extra='model_type=densenet121'

variable_key='select_depths'
variable_val=(18 20 22)
variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_'$variable_key'_'$index'/NCCT'

echo $variable';'$prefix';'$extra
export PYTHONPATH=$PYTHONPATH:$pwd
python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
