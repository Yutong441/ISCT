#modified from https://github.com/emayerhofer/ct_ml/blob/master/data_preparation/02%20prepare_images.ipynb
import functools, os, re, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import ipywidgets as widgets
import scipy.io
import skimage
import data_raw.CTname as CTN

def get3D (dirname, filename):
    view_name = CTN.get_basename (dirname)
    CTN.makedir ('./tmp')
    CTN.bash_in_python ('dcm2niix -z y -v n -f "{}"_%t_%s -o {} {}'.format (
        view_name, 'tmp', dirname))

    img = CTN.load_nib ('./tmp/'+re.sub ('.json$', '.nii.gz', filename))
    shutil.rmtree ('./tmp')
    return img

def show_levels (img, N=None, rows=None, title=None, windows=[0,100],
        overlay=None, fig_dir=None):
    '''
    Args:
        `img`: [height, width, depth]
        `N`: number of slices to show; default is all slices
        `rows`: number of rows of subplots
    '''
    if N is None: N = np.clip(img.shape[2], 0, 50)
    if rows is None: rows= int (np.ceil (np.sqrt(N)))
    stepsize = int(img.shape[2]//N)
    length = int (np.ceil (N/rows))
    fig, ax = plt.subplots (rows, length, figsize=(length*4.8, rows*4.8),
            squeeze=False)
    image = img.clip (windows[0], windows[1])
    if overlay is not None:
        image= skimage.color.label2rgb(overlay, image=image/windows[1], 
                bg_label=0, alpha=0.5)
    for i, level in enumerate (range (0, image.shape[2], stepsize)):
        if i < N:
            if len(img.shape)==3:
                ax[i//length, i%length].imshow (image[:,:,level], cmap='gray')
            elif len (img.shape)==4 and img.shape[3]==3:
                ax[i//length, i%length].imshow (image[:,:,level])

    if title is not None: fig.suptitle (title)
    [axi.set_axis_off () for axi in ax.ravel ()]
    if fig_dir is not None:
        fig.savefig (fig_dir, bbox_inches='tight',dpi=400)

def show_ori_dcm (filename, df_path, *args, **kwargs):
    df = pd.read_csv (df_path)
    df_index = df [df['anoID']==filename]
    dirname = df_index['directory'].iloc[0]
    filename = df_index ['out_file'].iloc[0]
    img = get3D (dirname, filename)
    show_levels (img, *args, **kwargs)

def callback_f (c, df, index, save_path):
    if c is True: 
        if index is not None:
            df[index:(index+1)].to_csv (save_path, mode='a', header=False,
                    index=False)
        else:
            entry = df[0:1]
            entry ['directory'].iloc[0] = 'NA_' + \
                    entry ['patientID'].iloc[0]
            entry.to_csv (save_path, mode='a', header=False, index=False)

    else: #erase the previously selected image
        save_csv = pd.read_csv(save_path, index_col=None)
        if index is not None: remove_dir = df['directory'].iloc[index]
        else: remove_dir = 'NA_'+df['patientID'].iloc[0]

        if remove_dir in save_csv['directory'].values:
            save_csv = save_csv[save_csv['directory']!=remove_dir]
            save_csv.to_csv (save_path, mode='w', header=True, index=False)

def widget_wrap_show_func (df, index, save_path, show_func=show_levels,
        callback=callback_f, disp=print):
    '''
    Wrap an image display function with checkbox widget
    Args:
        `df`: dataframe that contains the image metadata
        `index`: index of the row in `df` indicating which image to display
        `show_func`: a function that displays images
        `callback`: a function that is executed when the checkbox is ticked
        `disp`: a function that shows text, in jupyter notebook, this would be
        `IPython.display.display`
    '''
    if index is not None:
        dir_path = df['directory'].iloc[index]
        view_name=os.path.basename (re.sub('/$', '', dir_path))
        img = get3D (dir_path, df['out_file'].iloc[index])
        disp ('for {}, max {}, min {}, size {}'.format (view_name, 
            img.max(), img.min(), img.shape))
        show_func (img, title=view_name)
    else: disp ('no suitable images')

    callback_func = functools.partial (callback, df=df, index=index,
            save_path=save_path)
    cbox = widgets.Checkbox( value=False,
        description='This is the desired option',
        disabled=False, indent=False
    )
    disp(widgets.interactive_output(callback_func, {'c': cbox}))
    disp(cbox)

def show_multi_images (df_path, save_path, **kwargs):
    if 'disp' in kwargs.keys(): disp = kwargs ['disp']
    else: disp = print

    all_df = pd.read_csv(df_path, index_col=[0])
    # identify which cases have been viewed
    if not os.path.exists (save_path):
        empty_df = pd.DataFrame (columns=all_df.columns)
        empty_df.to_csv (save_path, index=False)
        list_viewed = np.array ([])
    else: list_viewed= np.unique (pd.read_csv (save_path)['patientID'])

    # obtain the first case to be viewed
    list_unviewed = all_df[~all_df['patientID'].isin (list_viewed)]
    if len (list_unviewed) != 0:
        selected_ID = list_unviewed['patientID'].iloc[0]
        sel_df = all_df [all_df['patientID']==selected_ID]

        num_unviewed = len (np.unique (list_unviewed['patientID']) )
        disp ('{} done, {} to do'.format (len(list_viewed), 
            num_unviewed ))
        disp ('Please select '+selected_ID)
        disp (sel_df [['view', 'thick', 'num']])

        for i in range(len (sel_df)):
            widget_wrap_show_func (sel_df, i, save_path, **kwargs)
        widget_wrap_show_func (sel_df, None, save_path, **kwargs)
    else: disp ('Done')

def multi_imshow_widget (*args, **kwargs):
    '''
    Visually select the appropriate images from one particular case or patient
    Args:
        `df_path`: path to the dataframe containing image directory
        (`directory` column) and patient/case/incident ID (`patientID` column).
        `directory` column are relative).
        `save_path`: where to save the selected images.
        `disp`: function to display text-based output, e.g., `IPython.display.display`
    Output:
        all images of a particular case
        checkbox widget alongside each image
        checkbox widget at the end to prompt moving to the next case
    '''
    cbox_next =widgets.Checkbox(value=False, description="next case",
            disabled=False, indent=False)
    def wrap_multi (b):
        if b is True: show_multi_images (*args, **kwargs)

    if 'disp' in kwargs.keys(): disp = kwargs ['disp']
    else: disp = print
    disp (widgets.interactive_output(wrap_multi, {"b": cbox_next}))
    disp (cbox_next)
