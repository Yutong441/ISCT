import re
from functools import partial
from torch import nn
from CNN.resnet_2d import ResNetResidualBlock
import CNN.CNN_basic as CB
import CNN.building_block as BB
import CNN.decoder as DE

class res3d (ResNetResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=2,
            op2d=False, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if op2d:
            conv=partial (CB.Conv2d_in_3d, out_chan=out_channels, kernel=3)
        else:
            conv=partial (CB.Conv3dAuto, out_channels=out_channels, 
                    kernel_size=3)
        self.expansion, self.downsampling = expansion, downsampling
        N = self.expanded_channels

        self.shortcut = nn.Sequential(
            conv(self.in_channels, stride=self.downsampling, bias=False),
            nn.BatchNorm3d(N)) if self.should_apply_shortcut else None

        block = [ conv(in_channels, stride=self.downsampling), 
                nn.BatchNorm3d( N), nn.ReLU (), conv(N, stride=1),
                nn.BatchNorm3d (N) ]
        self.blocks = nn.Sequential (*block)

class CNN3d (nn.Module):
    def __init__(self, in_channels, n_classes, extra_features=0,
            add_sigmoid='none', times_max=1, init_chan=16, layer=5,
            n_decoder=1, step=4, *args, **kwargs):
        super().__init__()
        self.encoder = CB.CNN3d_encoder (in_channels, layer=layer,
                init_chan=init_chan, *args, **kwargs)
        final_chan = init_chan*(2**(layer-1))

        out_decoder = CB.CNN3d_decoder (final_chan, n_classes,
                n_decoder=n_decoder, step=step)
        if extra_features >0:
            self.decoder = DE.fuse_decoder (n_classes=n_classes,
                decoder1= out_decoder, 
                decoder2 = DE.lin_decoder (extra_features))
        else: self.decoder= out_decoder

        self.add_sigmoid = add_sigmoid
        self.times_max = times_max

    def forward (self, x):
        if type(x)!=tuple and type(x) != list: inp = x
        else: inp = x[0]
        encode_x = self.encoder (inp)

        if type(x)!=tuple and type(x) != list: out = encode_x
        else: out = [encode_x, x[1]]
        out = self.decoder(out)

        out = BB.act_func (out, self.add_sigmoid, self.times_max)
        return out

def CNN3d_n (in_channels, n_classes, model_type, *args, **kwargs):
    attention = model_type.split('_')[1]
    layer = int (re.sub ('^CNN', '', model_type.split('_')[0]))
    return CNN3d (in_channels, n_classes, layer=layer, attention=attention,
            *args, **kwargs)
