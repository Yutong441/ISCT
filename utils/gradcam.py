import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from medcam import medcam
from data_raw.CTname import load_nib 
from CNN.resnet_half import resCRNN_perturb 

def save_gradcam (model, loader_set, cfg, save_dir=None, mode='test'):
    if save_dir is None:
        save_dir = cfg ['save_prefix'] + '_gradcam'
        if not os.path.exists (save_dir): os.mkdir (save_dir)

    model.eval ()
    ytrue = loader_set.annotations [cfg ['outcome_col']].values
    ypred = np.load (cfg['save_prefix']+'_tmp/ypred_{}.npz'.format (
        mode)) ['arr_0']

    model = medcam.inject(model, backend="gcam", layer=cfg['gcam'],
            save_maps=False, label='best', replace=True)
    torch.backends.cudnn.enabled=False
    for i in range (loader_set.__len__ () ):
        img, lab = loader_set.__getitem__ (i)
        act = model (img.unsqueeze (0).to(cfg['device']))
        save_obj = {'map': act.detach().cpu().numpy(),
                'ypred': float (ypred[i]), 'ytrue': float (ytrue[i])}
        filename = loader_set.annotations.index[i]
        np.savez_compressed (save_dir+'/cam_'+filename, save_obj)

def central_slice (img, depths=21):
    start_depth = (img.shape[2] - depths)//2
    return img [:,:, start_depth:(start_depth + depths)]

class IndexTracker:
    def __init__(self, ax, X, Y, cmap_type='jet', title=None):
        self.ax = ax
        if title is not None: ax.set_title (title)
        self.X, self.Y = X, Y
        rows, cols, self.slices = Y.shape
        self.ind = self.slices//2
        self.im1 = ax.imshow(self.X[:, :, self.ind], cmap=cmap_type)
        self.im2 = ax.imshow(self.Y[:, :, self.ind], cmap='viridis', alpha=0.5)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up': self.ind = (self.ind + 1) % self.slices
        else: self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im1.set_data(self.X[:, :, self.ind])
        self.im2.set_data(self.Y[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind, fontname='Arial')
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()

class MultiIndexTracker:
    def __init__(self, ax, X, Y, maps, ncols=4, title=None, cmap_types=None):
        [one_ax.axes.xaxis.set_ticks ([]) for one_ax in ax.ravel()]
        [one_ax.axes.yaxis.set_ticks ([]) for one_ax in ax.ravel()]
        [one_ax.spines['top'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['bottom'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['right'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['left'].set_visible(False) for one_ax in ax.ravel()]
        self.ax = ax

        self.X, self.Y = X, Y
        rows, cols, self.slices = Y.shape
        self.ind = self.slices//2
        self.list1, self.list2 = [], []
        self.ncols = ncols
        self.maps = maps

        if cmap_types is None:
            cmap_types = ['gray'] + ['gnuplot2']*2 + ['gist_rainbow_r']*3 + \
                    ['gnuplot2']
        elif len (cmap_types) == 1: cmap_types = [cmap_types]*len (maps)
        assert len (cmap_types) == len (maps), \
            'The number of cmap should match the number of channels'

        for i in range (X.shape [-1]):
            cmap_type = mpl.cm.get_cmap(cmap_types[i]).copy()
            cmap_type.set_under(color='black') 

            vmax = X [...,i].max()
            self.ax [i//ncols, i%ncols].set_title (maps[i])
            self.list1.append (self.ax [i//ncols, i%ncols].imshow(
                    self.X[:, :, self.ind, i], cmap=cmap_type, vmin=1e-2,
                    vmax=vmax))
            if maps [i] == 'MIP':
                self.list2.append (self.ax [i//ncols, i%ncols].imshow(
                        self.Y[:, :, self.ind], cmap='viridis', alpha=0.5))
            else: self.list2.append (None)
            cbar = plt.colorbar (self.list1[i], ax=self.ax [i//ncols, i%ncols],
                    shrink=0.7)
            cbar.set_ticks([])

        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up': self.ind = (self.ind + 1) % self.slices
        else: self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        ncols = self.ncols
        for i in range (self.X.shape[-1]):
            self.list1[i].set_data(self.X[:, :, self.ind, i])
            self.ax [i//ncols, i%ncols].set_ylabel('slice %s' % self.ind)
            self.list1[i].axes.figure.canvas.draw()
            if self.maps [i] == 'MIP':
                self.list2[i].set_data(self.Y[:, :, self.ind])
                self.list2[i].axes.figure.canvas.draw()

def multi_plot3D (img, overlay, ncols=4, channel_name=None, plot_title=None):
    '''
    Args:
        `img`: [H, W, D, C]
        `overlay`: [H, W, D]
    '''
    chan = img.shape[-1]
    nrows = int (np.ceil (chan/ncols))
    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    if channel_name is None:
        maps = ['MIP', 'CBF', 'CBV', 'TTP', 'TTD', 'MTT', 'PMB']
    tracker = MultiIndexTracker(ax, img, overlay, maps)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    if plot_title is not None: plt.suptitle (plot_title)
    plt.show ()

def load_data (sample_num='001', img_path='data/CTP/CTP_pro/CTP', 
        overlay_path= 'results/CTP_gradcam/cam_CTP'):
    obj = np.load (overlay_path+sample_num+'.npz', allow_pickle=True)
    plot_title = 'predicted: {}, truth: {}'.format (
          np.round (obj['arr_0'].item()['ypred'], 1), 
          obj['arr_0'].item()['ytrue'])
    act = obj['arr_0'].item()['map'].squeeze()
    act = np.nansum (act, axis=1)
    act = np.moveaxis (act, 0, 2)
    img = central_slice (load_nib (img_path+sample_num+'.nii.gz'))
    return act, img, plot_title

def load_plot_3D (sample_num='001', img_path='data/CTP/CTP_pro/CTP', 
        overlay_path= 'results/CTP_gradcam/cam_CTP', ncols=4,
        channel_name=None):
    act, img, plot_title = load_data (sample_num, img_path, overlay_path)
    multi_plot3D (img, act, ncols=ncols, channel_name=channel_name,
            plot_title=plot_title)

def perturb_img (loader_set, device, cg):
    model = resCRNN_perturb (**cg.get_model_args (cg.config))
    model.load_state_dict(torch.load(cg.config['save_prefix']+'_model', 
        map_location=torch.device (device)))
    model.eval ()

    save_dir = cg.config ['save_prefix']+'_perturb'
    if not os.path.exists (save_dir): os.mkdir (save_dir)
    with torch.no_grad ():
        for i in range (loader_set.__len__ () ):
            img, lab = loader_set.__getitem__ (i)
            ID = loader_set.annotations.index[i]
            model.embed (img.unsqueeze (0))
            perturbation = model.perturb_vol ()
            np.savez_compressed (cg.config ['save_prefix']+'_perturb/per_'+ID,
                    perturbation)

def load_perturb_plot (num='006', fea_num=100, img_path='data/CTP/CTP_pro/CTP',
        overlay_path= 'results/CTP_gru2_reg/CTP_perturb/per_CTP'):
    img = central_slice (load_nib (img_path+num+'.nii.gz'))
    act = np.load (overlay_path+num+'.npz')['arr_0']
    overlay = F.interpolate (torch.tensor (act).permute(3,0,1,2), 
            img.shape[:2], mode='bilinear')
    overlay_np = overlay[fea_num].permute (1,2,0).numpy ()
    multi_plot3D (img, abs (overlay_np))
