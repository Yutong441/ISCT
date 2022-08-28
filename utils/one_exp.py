'''
Run one experiment of CNN.
Non-default configs are supplied through the `--config` argument
'''
import utils.config as cg
from utils.train_utils import train_test
from utils.test_utils import test_model
import argparse

# dataset features
base_cf='data_folder=[tmp_images/reg_trim,tmp_images/reg_trim];'
base_cf+='label_dir=[labels_LSW/val.csv,labels_LSW/test.csv];'
base_cf+='select_depths=18; common_shape=[130,170]; transform=None;'

# training features
base_cf+='batch_size=16; lr=0.0001; outcome_col=LSW; pos_label=4.5;'

# model architecture
base_cf+='input_channels=1; predict_class=1; model_type=densenet169;'
base_cf+='loss_type=regression; add_sigmoid=None; times_max=1;'
base_cf+='attention=None;'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    cg.config = cg.sub_dict (cg.config, base_cf+args.config)
    train_test (cg)
    test_model (cg)
