# Does image registration improves performance accuracy?
index=${1:-0}
prefix='dense'
extra='model_type=densenet121'

variable_key='common_shape'
variable_val=('[tmp_images/reg_trim,tmp_images/reg_trim]' '[tmp_images/unreg_trim,tmp_images/unreg_trim]' )

variable=$variable_key'='${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_'$variable_key'_'$index'/NCCT'

python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
