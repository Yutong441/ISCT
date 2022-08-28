# impact of batch size on model performance
index=${1:-1}
prefix='dense'
extra='model_type=densenet121'

variable_key='batch_size'
variable_val=(16 12 8)
variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix='$prefix'_'$variable_key'_'$index

python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
