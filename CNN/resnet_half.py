# break resCRNN into 2 halves to investigate the meaning of deep features
import torch
from CNN.resRNN import resCRNN, RNN_decoder
from CNN.resnet_2d import ResNetBasicBlock

class RNN_features (RNN_decoder):
    def __init__ (self, in_channels, n_classes, n_decoder=2, step=4, dropout=0):
        super (RNN_features, self).__init__ (in_channels, n_classes, n_decoder,
                step, dropout)

    def forward (self, x, batch_size):
        '''return the rnn layer output, not the final output via the linear
        layer'''
        avg = self.avg (x) # [L*B, C, 1, 1]
        LxB, C, _, _ = avg.shape
        L = LxB//batch_size
        rnn_out =  self.rnn (avg.reshape(L, batch_size, C)) #L, B, C
        return rnn_out[0] [-1]

class resCRNN_perturb (resCRNN):
    def __init__(self, in_channels, n_classes, ampli=1, add_sigmoid=False,
            times_max=1, n_decoder=2, step=4, dropout=0, deepths=[2,2,2,2],
            block=ResNetBasicBlock, *args, **kwargs):
        super(resCRNN_perturb, self).__init__(in_channels, n_classes,
            add_sigmoid=False, times_max=1, n_decoder=n_decoder, step=step,
            dropout=dropout, deepths=deepths, block=block, *args, **kwargs)
        self.ampli = ampli
        self.decoder = RNN_features (512, n_classes, n_decoder, step,
                dropout)
        
    def embed (self, x):
        ''' 
        Args:
            `x`: [1, C, D, H, W], can only accept one image at a time
        Return:
            `self.encoder_x`: [D, C, H, W], the embedding by the encoding blocks
        '''
        B, C, D, H, W = x.shape
        assert B == 1, 'the function can only process one image at a time'
        x = x.permute (2,0,1,3,4) # [D, 1, C, H, W]
        self.encoder_x = self.encoder(x.reshape(-1, C, H, W) )
        self.rnn_val = self.decoder (self.encoder_x, 1)
        print ('finished embedding the image')

    def perturb_pos (self, i, k):
        ''' Amplify the value at a location (i, :, k) '''
        D, C, H, W = self.encoder_x.shape
        device = 'cuda:0' if torch.cuda.is_available () else 'cpu'
        ampli_encode = torch.zeros([W, D, 1, H, W], device=device)
        max_en = self.encoder_x.max()
        for j in range (W): ampli_encode [j,k,0,i,j] = self.ampli*max_en
        encode_inp = (self.encoder_x.unsqueeze(0)+ampli_encode).permute (1,0,2,3,4)
        rnn_out = self.decoder(encode_inp.reshape(-1, C, H, W), W) # B, C
        return rnn_out - self.rnn_val 

    def perturb_slice (self, k):
        D, C, H, W = self.encoder_x.shape
        return torch.stack ([self.perturb_pos (i, k) for i in range (H)], dim=0)

    def perturb_vol (self): 
        D, C, H, W = self.encoder_x.shape
        return torch.stack ([self.perturb_slice (k) for k in range (D)], dim=0)
