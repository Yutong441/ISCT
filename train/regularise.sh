index=${1:-0}
prefix=''
extra=''

variable_val=('L1=0.01' 'L1=0.001' 'L2=0.01' 'L2=0.0001' 'dropout=0.1' 'dropout=0.01')

variable=${variable_val[$index]}
prefix='save_prefix=results/'$prefix'_regularise_'$index'/NCCT'

export PYTHONPATH=$PYTHONPATH:$pwd
python3 utils/one_exp.py --config=$variable';'$prefix';'$extra
