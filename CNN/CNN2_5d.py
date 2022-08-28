import torch
from torch import nn
import torch.nn.functional as F

class Conv2d_in_3d (nn.Conv3d):
    def __init__(self, in_chan, out_chan, kernel=3, stride=2, *args, **kwargs):
        super().__init__(in_chan, out_chan, kernel_size= (1, kernel, kernel), 
                stride=(1, stride, stride), *args, **kwargs)
        self.padding =  (0, kernel//2, kernel//2) 
        # dynamic add padding based on the kernel_size

class Conv1d_in_3d (nn.Conv3d):
    def __init__(self, in_chan, out_chan, kernel=3, stride=1, *args, **kwargs):
        super().__init__(in_chan, out_chan, kernel_size= (kernel, 1, 1), 
                stride=1, *args, **kwargs)
        self.padding =  (kernel//2, 0, 0) 

def conv_block (in_chan, out_chan, add_1d=True):
    block=[Conv2d_in_3d (in_chan, out_chan, kernel=3)]
    if add_1d:
        block.extend (
            [Conv1d_in_3d (out_chan, out_chan, kernel=3), 
            nn.MaxPool3d (kernel_size=(2,1,1))])
    block.extend ([nn.BatchNorm3d (out_chan), nn.ReLU () ])
    return block

class CNN2_5d_decoder (nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        #global max pooling along depth
        if type(x)!=tuple and type(x) != list:
            x = F.max_pool3d (x, kernel_size=(x.size()[2], 1, 1))
            x = self.avg(x) 
            x = x.view(x.size(0), -1)
            x = self.decoder(x)
            return x
        else:
            x_enc = F.max_pool3d (x[0], kernel_size=(x[0].size()[2], 1, 1))
            x_enc = self.avg(x_enc) 
            x_enc = x_enc.view(x_enc.size(0), -1)
            x = self.decoder(torch.cat ([x_enc, x[1]], axis=1))
            return x

class CNN2_5d (nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        block = [*conv_block (in_channels, 32), 
                *conv_block (32,64),
                *conv_block (64,128, add_1d=False),
                *conv_block (128,256, add_1d=False),
            ]
        self.blocks = nn.Sequential (*block)
        self.decoder = CNN2_5d_decoder (256, n_classes)

    def forward (self, x):
        x = self.blocks (x)
        return self.decoder (x)
