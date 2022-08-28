import re
import numpy as np

config = {}
# directory
config ['root']='data/'
config ['save_prefix']='results/DWI/DWI'
config ['archive'] = 'logs/'
#directory and prefix for all the files to be saved, including the model
#weight, loss history and metric history

# learning rate
config ['num_epochs'] = 150
config ['lr'] = 0.005
config ['L2'] = 0
config ['L1'] = 0
config ['regularised_layers'] = []
config ['step_size']=10
config ['gamma']=0.5
config ['resume_training']= False
config ['pretrained'] = False
config ['train_weight'] = True
config ['old_save_path'] = 'None'

# data
config ['data_folder'] = [''] 
#must be a list, if multiple paths are supplied, the first one is assumed to
#be the training/internal validation ones, all others are external validation

config ['label_dir'] = ['labels']
# similar to config ['data_folder'], each label must correspond to the data

config ['select_channels'] = None
config ['select_depths'] = 19
config ['select_num'] = None
config ['num_workers'] = 2
config ['pin_memory'] = True
config ['batch_size'] = 4
config ['device'] = 'cuda:0' #'cpu' or 'cuda:0'
config ['common_shape'] = [256, 256]
config ['downsize'] = None
config ['transform'] = 'default'

# model
config ['input_channels'] = 6
config ['predict_class'] = 6
config ['add_sigmoid'] = 'none'
config ['times_max'] = 1
config ['model_type'] = 'resnet18'
config ['loss_type'] = 'classification'
config ['sigma'] = 0.5 # optional
config ['prob_fun'] = 'sigmoid' #optional; can be 'sigmoid' or 'gauss'
config ['decoder'] = '0conv_2lin_0pool'
config ['dropout'] = 0
config ['step_linear'] = 2
config ['add_STN'] = False
config ['init_chan'] = 16
config ['output_features'] = None

# evaluation
config ['outcome_col'] = 'mRS'
config ['index_col'] = 'anoID'
config ['all_metrics'] = ['AUC', 'MSE', 'accuracy', 'sensitivity', 'specificity']
config ['eval_every'] = 2
config ['initial_metric'] = 0
config ['metric'] = 'AUC'
config ['tolerance'] = 20
config ['better'] = 'pos' #either 'pos' or 'neg'
config ['pos_label'] = 2

# visualise
config ['gcam'] = 'encoder.blocks.3' #'features.32' for vgg13_bn
config ['save_gradcam'] = False
config ['level'] = 10
config ['show_num'] = 3
config ['x_features'] = ['Age', 'Sex']
config ['y_features'] = ['mRS_dc']

def get_data_args(config):
    assert len(config ['data_folder']) == len (config ['label_dir'])
    loader_arg = [{
            'data_folder': config ['data_folder'][i],
            'select_channels': config ['select_channels'],
            'select_depths': config ['select_depths'],
            'select_num': config ['select_num'],
            'outcome_col': config ['outcome_col'],
            'index_col': config ['index_col'],
            'common_shape': config ['common_shape'],
            'output_features': config ['output_features'],
            'label_dir': config ['label_dir'][i]
    } for i in range(len(config ['data_folder']))]

    data_arg = {
            'num_workers': config ['num_workers'],
            'pin_memory': config ['pin_memory'],
            'batch_size': config ['batch_size']
    }
    return loader_arg, data_arg

def get_model_args (config):
    if 'cum_link' in config ['loss_type']: n_class = 1
    else:
        if 'ordinal' in config ['loss_type']: 
            n_class = config ['predict_class'] - 1
        else: n_class = config ['predict_class'] 
    n_decode = 0 if config ['output_features'] is None else len (
            config ['output_features'])
    resblock = True if config ['model_type'] == 'resunet' else False
    n_conv = int(re.sub ('conv$', '',config ['decoder'].split ('_')[0]))
    n_decoder = int(re.sub ('lin$', '',config ['decoder'].split ('_')[1]))
    if len (config ['decoder'].split ('_')) >1:
        n_pool = int(re.sub ('pool$', '',config ['decoder'].split ('_')[2]))

    return {'in_channels': config ['input_channels'], 
        'n_classes': n_class,
        'model_type': config ['model_type'], 
        'add_sigmoid': config ['add_sigmoid'],
        'times_max': config ['times_max'],
        'coral': 'coral' in config ['loss_type'],
        'extra_features': n_decode,
        'n_conv': n_conv,
        'n_decoder': n_decoder,
        'dropout': config ['dropout'],
        'step': config ['step_linear'],
        'resblock': resblock,
        'n_pool': n_pool,
        'add_STN': config ['add_STN'],
        'init_chan': config ['init_chan'],
        'img_shape': [config ['select_depths']] +config ['common_shape'],
        'two_stage': config ['loss_type']=='two_stage'}

def add_entry (explain, val):
    '''
    Add entry to printing config
    Args:
        `explain`: explanatory text accompanying a parameter, must contain '{}'
        somewhere
        `val`: the parameter value
    '''
    if type (val) == np.ndarray or type (val) == list:
        txt = ', '.join ([str(i) for i in val])
    else: txt = val
    return explain.format (txt)

def trim_ws (xx):
    out = re.sub (' +$', '', xx)
    out = re.sub ('^ +', '', out)
    return out

def convert (xx):
    out = trim_ws (xx)
    try: 
        num_out = float (out)
        if re.search ('\.', out) is None: out = int (out)
        else: out = num_out
    except:
        front = re.search('^(\[|\{)', out)
        back = re.search('(\]|\})$', out)
        # generate list
        if front is not None and back is not None:
            out = re.sub ('(\[|\{|\}|\])', '', out)
            out = out.split (',')
            try: out = [float(i) for i in out]
            except: out = out
        elif xx == 'None': out = None
        else: out = xx
    return out

def str_to_dict (dict_str, item_sep=';', key_sep='='):
    dict_list = dict_str.split (item_sep)
    return {trim_ws(i.split(key_sep)[0]):convert (i.split(key_sep)[1]
        ) for i in dict_list}

def sub_dict (old_dict, dict_str, *args, **kwargs):
    new_dict = str_to_dict (dict_str, *args, **kwargs)
    for key, val in new_dict.items():
        old_dict [key] = val
    return old_dict

def print_config (cf, join_str=True):
    str_list = []
    str_list.append (add_entry ('where the data is stored: {}', cf['root']))
    str_list.append (add_entry ('where the results are saved: {}',
        cf['save_prefix']))

    str_list.append ('')
    str_list.append ('# learning rate')
    str_list.append (add_entry ('number of training epochs: {}',cf ['num_epochs']))
    str_list.append (add_entry ('initial learning rate: {}',cf['lr']))
    str_list.append (add_entry ('L2 weight decay: {}',cf['L2']))
    str_list.append (add_entry ('L1 weight decay: {}',cf['L1']))
    str_list.append ('multiply the learning rate by {} every {} steps'.format
            (cf ['gamma'], cf ['step_size']))
    str_list.append (add_entry ('whether to resume training from the last saved model weights: {}',
        cf ['resume_training']))
    str_list.append (add_entry ('load previous model: {}',cf ['old_save_path']))

    str_list.append ('')
    str_list.append ('# transfer learning')
    str_list.append (add_entry ('whether to initialise with pretrained resnet18: {}',
        cf ['pretrained']))
    str_list.append (add_entry ('whether to train the pretrained weights: {}',
        cf ['train_weight']))

    str_list.append ('')
    str_list.append ('# data')
    str_list.append (add_entry ('Imaging data are stored in: {}', 
        [cf ['root'] +'/' + i for i in cf ['data_folder']] ))
    str_list.append (add_entry ('Labels are stored in: {}', 
        [cf ['root'] +'/' + i for i in cf ['label_dir']] ))
    str_list.append (add_entry ('number of channels selected from the original image: {}', 
        cf ['select_channels'] ))
    str_list.append (add_entry ('the central {} slices selected', 
        cf ['select_depths']))
    str_list.append (add_entry ('number of images selected for training and testing: {}', 
        cf['select_num']))
    str_list.append (add_entry ('number of workers to prepare images in CPU: {}',
        cf ['num_workers']))
    str_list.append (add_entry ('batch size: {}', cf ['batch_size']))
    str_list.append (add_entry ('the model is trained on: {}',cf ['device']))
    str_list.append (add_entry ('Images are padded to: {}',cf['common_shape']))
    str_list.append (add_entry ('Images are downsampled to: {}',cf['downsize']))

    str_list.append ('')
    str_list.append ('# model')
    str_list.append (add_entry ('number of input channels to CNN: {}',
        cf['input_channels']))
    str_list.append (add_entry ('number of predicted classes: {}',
        cf ['predict_class']))
    str_list.append (add_entry ('type of model: {}', cf ['model_type']))
    str_list.append (add_entry ('type of loss: {}', cf ['loss_type']))
    if cf ['loss_type'] == 'KL':
        str_list.append (add_entry ('sigma for KL is: {}', cf ['sigma']))
    if cf ['loss_type'] == 'cum_link':
        str_list.append (add_entry ('probability distribution is: {}', 
            cf ['prob_fun']))
    str_list.append (add_entry ('whether to add sigmoid: {}',
        cf['add_sigmoid']))
    str_list.append (add_entry ('sigmoid output multiplied by: {}',
        cf['times_max']))
    str_list.append (add_entry ('organisation of the decoding layers: {}',
        cf['decoder']))
    str_list.append (add_entry ('dropout among linear layers: {}',
        cf['dropout']))
    str_list.append (add_entry ('Reducing feature number by {} in linear layers',
        cf['step_linear']))
    str_list.append (add_entry ('STN layer added: {}',
        cf['add_STN']))
    str_list.append (add_entry ('initial channel after the first layer: {}',
        cf['init_chan']))

    str_list.append ('')
    str_list.append ('# metrics')
    str_list.append (add_entry ('the target outcome is: {}', 
        cf ['outcome_col']))
    str_list.append (add_entry ('filename is located in: {}', 
        cf ['index_col']))
    str_list.append (add_entry ('the tested metrics: {}', cf ['all_metrics']))
    str_list.append (add_entry ('evaluate the metrics every {} epoch', 
        cf ['eval_every']))
    str_list.append (add_entry ('expected initial value for the metric of interest: {}',
        cf ['initial_metric']))
    str_list.append (add_entry ('on which metric to determine model improvement: {}',
        cf ['metric']))
    str_list.append (add_entry ('after {} epochs of failure of improvement the training will stop',
        cf ['tolerance']))
    str_list.append (add_entry ('whether a higher value in the metric of interest is better: {}',
        cf ['better']== 'pos'))
    str_list.append (add_entry ('predicted value equal to or above {} is classified as positive',
        cf ['pos_label']))

    str_list.append ('')
    str_list.append ('# visualise')
    str_list.append (add_entry ('the level of imaging plane in display: {}',
        cf ['level']))
    str_list.append (add_entry ('number of images to show: {}',cf ['show_num']))
    str_list.append (add_entry ('the layer from which activation map is visualised: {}',
        cf ['gcam']))
    str_list.append (add_entry ('the non-imaging features for DR are: {}',
        cf ['x_features']))
    str_list.append (add_entry ('the non-imaging features for visualisation are: {}',
        cf ['y_features']))
    str_list.append (add_entry ('the non-imaging features for neural net are: {}', 
        cf['output_features']))
    if join_str: return '\n'.join (str_list)
    else: return str_list
