# impact of learning rate on model performance
index=${1:-1}
prefix='dense'
extra='model_type=densenet121'

variable_key='lr'
variable_val=(0.0001 0.001 0.01)
variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_'$variable_key'_'$index'/NCCT'

echo $variable';'$prefix';'$extra
export PYTHONPATH=$PYTHONPATH:$pwd
python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
