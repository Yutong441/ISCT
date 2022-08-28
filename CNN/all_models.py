import re
import utils.config as cg

def choose_models (cfg):
    if 'CNN' in cfg ['model_type']:
        from CNN.CNN3d import CNN3d
        return CNN3d (**cg.get_model_args (cfg))
    elif 'resCRNN' in cfg ['model_type']: 
        from CNN.resRNN import resCRNN_n 
        return resCRNN_n (**cg.get_model_args (cfg))
    elif 'resnet' in cfg ['model_type']:
        from CNN.resnet_3d import resnet3d_n
        return resnet3d_n (**cg.get_model_args (cfg))
    elif 'densenet' in cfg['model_type']:
        from CNN.densenet import densenet_n 
        return densenet_n (**cg.get_model_args (cfg))

