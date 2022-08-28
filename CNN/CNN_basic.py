import torch
from torch import nn
import torch.nn.functional as F
from CNN.STN import STN
import CNN.building_block as BB
import CNN.decoder as DE
        
def conv_block (in_chan, out_chan, op2d=False):
    if op2d:
        block=[BB.Conv2d_in_3d (in_chan, out_chan, kernel=3)]
    else:
        block=[BB.Conv3dAuto (in_chan, out_chan, kernel_size=3, stride=2)]
    block.extend ([nn.BatchNorm3d (out_chan), nn.ReLU () ])
    return block

class CNN3d_encoder (nn.Module):
    def __init__(self, in_channels, attention=None, add_STN=False, layer=5,
            init_chan=16, img_shape=(18, 128, 128), *args, **kwargs):
        super().__init__()
        assert layer >2, 'the number of CNN layers must be at least 2'
        # the first 2 layers perform 3D convolution
        block = [*conv_block (in_channels, init_chan, op2d=False), 
                *conv_block (init_chan, init_chan*2, op2d=False)]

        # the later layers perform 2D convolution, because there would not be
        # enough depths
        for i in range(2, layer):
            expand = 2**(i-1)
            block.extend([*conv_block (init_chan*expand,
                init_chan*expand*2, op2d=True)])

        final_chan = init_chan*(2**(layer-1))
        atten = BB.attention_block (attention, final_chan)
        if atten is not None: block.append (atten)

        if add_STN: block = [STN (in_channels, img_shape)] + block
        self.blocks = nn.Sequential (*block)

    def forward (self, x):
        return self.blocks(x)

class CNN3d_decoder (DE.CNN_decoder):
    def __init__(self, in_channels, n_classes, step=4, n_decoder=2):
        embed_chan = in_channels//(step**(n_decoder-1))
        super().__init__(embed_chan=embed_chan)

        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.hidden = BB.multi_linear (in_channels, 
                layer =n_decoder-1, step=step)
        self.linear = nn.Linear(embed_chan, n_classes)

    def embedding (self, x):
        #global max pooling along depth
        x = F.max_pool3d (x, kernel_size=(x.size()[2], 1, 1))
        x = self.avg(x) 
        x = x.view(x.size(0), -1)
        x = self.hidden (x)
        return x
