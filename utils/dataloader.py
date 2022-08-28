import numpy as np
import pandas as pd
import os, re
import torch
import torch.nn.functional as F
from torchvision import transforms
from data_raw.CTname import load_nib

def padding_to_shape (x, to_shape):
    '''
    Args:
        `x`: pytorch tensor. The last two dimensions to be padded
        `to_shape`: the shape of the final padded tensor
    '''
    shape_diff = np.array (to_shape) - np.array (x.shape)[::-1][:2]
    pad_dim = [shape_diff[0]//2, shape_diff[0]//2+shape_diff[0]%2, 
            shape_diff[1]//2, shape_diff[1]//2+shape_diff[1]%2]
    return F.pad (x, pad_dim)

def random_depth_more (img, depths, random=True):
    D = img.shape[1]
    if random: start_depth = np.random.choice (D - depths, 1)[0]
    else: start_depth = (D - depths)//2 
    return img [:, start_depth:(start_depth + depths)]

def random_depth_less (img, depths, random=True):
    _, D, H, W = img.shape
    if random: bottom_num =np.random.choice (depths -D, 1)[0] 
    else: bottom_num = (depths -D )//2
    top_num = depths - D - bottom_num
    img_list = [torch.zeros ([1, top_num, H, W]), img,
            torch.zeros ([1, bottom_num, H, W])]
    return torch.cat (img_list, axis=1)

def random_depth (img, depths, random=True, max_depth=30):
    while img.shape[1] >= max_depth: img = img [:,::2]
    if img.shape[1] > depths: 
        return random_depth_more (img, depths, random=random)
    elif img.shape[1] == depths: return img
    else: return random_depth_less (img, depths, random=random)

def to_device (xx, device):
    if type (xx) not in [list, tuple]: return xx.to(device)
    else: return [i.to(device) for i in xx]

class dataloader (torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, data_folder, transformation='default',
            outcome_col='mRS', index_col='anoID', common_shape=None, 
            downsize=None, select_channels=None, select_depths=None,
            select_num=None, output_features=None, label_dir='labels'):
        '''
        Args:
            `root_dir`: top-level directory for a particular dataset, which
            should contains `train`, `test` and `labels` folder
            `mode`: 'train', 'test' or 'validate'
            `common_shape`: the height and width for all images. If an image
            does not have this shape, zero padding is used. 
            `select_channels`: which channels to be in the input
            `select_depths`: an integer, number of central slices to select
            `select_num`: number of images to select
            `output_features`: output other information along with the images
        '''
        self.root_dir = root_dir
        self.data_folder= data_folder
        self.common_shape = [int(i) for i in common_shape]

        if re.search ('.csv$', label_dir) is not None:
            if mode == 'train':
                label_dir = os.path.dirname (label_dir)
                label_path = root_dir+'/'+label_dir+'/train.csv'
            else: label_path=root_dir+'/'+label_dir
        else: label_path=root_dir+'/'+label_dir+'/'+ mode+'.csv'

        self.annotations = pd.read_csv(label_path)
        self.annotations.index = self.annotations[index_col]
        self.annotations = self.annotations [~pd.isna(
                self.annotations[outcome_col])]
        if select_num is not None:
            self.annotations = self.annotations [:select_num]

        self.ylabels = torch.tensor (self.annotations[outcome_col
            ].values)
        self.mode = mode
        self.channels = select_channels
        self.depths = select_depths
        if downsize is not None: self.downsize = [int(i) for i in downsize]
        else: self.downsize =None
        self.output_features = output_features

        if transformation == 'default':
            transformation = torch.nn.Sequential (
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine (0, translate=(0.05, 0.05),
                        scale=(0.95, 1.05)),
                    transforms.ColorJitter (brightness=0.1),
                    transforms.GaussianBlur (3)
                    )
            transformation = torch.jit.script (transformation)
        if mode != 'train': transformation = None
        self.transform = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.index[index]
        filename = self.root_dir+'/'+self.data_folder+'/'+img_id+'.nii.gz'
        img = load_nib (filename)
        img = torch.tensor (img)
        if (len (img.shape)==3): img = img.unsqueeze (3)

        # make sure every image has the same shape
        img = img.permute (3, 2, 0, 1)
        if self.common_shape != img.shape[2:]:
            img = padding_to_shape (img, self.common_shape)
        if self.downsize is not None: img = F.interpolate (img, self.downsize) 
        if self.channels is not None: img = img [self.channels]
        if self.depths is not None: 
            img = random_depth (img, self.depths, self.transform)

        # perform image augmentation
        if self.transform is not None:
            old_shape = img.shape
            new_shape = [-1, 1, *self.common_shape]
            img = self.transform(img.reshape(new_shape))
            img = img.reshape (old_shape)
            # randomly flip z axis
            if np.random.rand(1) > 0.5: img = torch.flip (img, dims=[1])
        if self.output_features is None:
            return (img.type (torch.float32), self.ylabels[index])
        else:
            output_df = self.annotations[self.output_features].iloc[index]
            return ((img.type (torch.float32), 
                    torch.tensor(output_df).type (torch.float32)),
                    self.ylabels[index])
