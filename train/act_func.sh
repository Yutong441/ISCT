# impact of batch size on model performance
index=${1:-1}
prefix='act'
extra='model_type=densenet121;add_sigmoid=tanh'

variable_key='times_max'
variable_val=(5 4.5 4 3.5)
variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_'$variable_key'_'$index'/NCCT'

python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
