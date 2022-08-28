# segment arterial territtory
import os, re, argparse
import numpy as np
import pandas as pd
import data_raw.CTname as CTN
import segmentation.bisect_brain as BB

def segment_territory (img, mask, window=[-100, 100]):
    img = img.clip (window[0], window[1])[...,np.newaxis]
    mask = (mask >0.5).astype(int)
    vol = (mask).sum(axis=(0,1,2))
    HU = (img*mask).sum(axis=(0,1,2))
    return np.array (list(vol)+list(HU/vol))

def choose_colnames (index):
    if index == 'tissue': #GM, WM or CSF
        quant_col = ['GM', 'WM', 'CSF']
        quant_col = [j+i for j in ['R_', 'L_'] for i in quant_col ]
    elif index == 'artery_level1':
        quant_col = ['ACA', 'MLS', 'LLS', 'MCAF', 'MCAP', 'MCAT', 'MCAO',
                'MCAI', 'PCAT', 'PCAO', 'PCTP', 'ACTP', 'BA', 'SC', 'IC', 'LV']
        quant_col = [j+i for i in quant_col for j in ['L_', 'R_'] ]

    elif index == 'artery_level2':
        quant_col = ['ACA2', 'MCA', 'PCA', 'VB', 'LV2']
        quant_col = [j+i for i in quant_col for j in ['L_', 'R_'] ]

    quant_col = [i+j for j in ['_vol', '_HU'] for i in quant_col]    
    return quant_col

def sum_images (img_dir, mask_dir, save_path, quant_col, quant_func):
    '''
    Summarise characteristics of a batch of images
    Args:
        `img_dir`: directory containing images to be analysed
        `mask_dir`: directory containing corresponding masks of the images. The
        names of the image and its corresponding mask must be the same.
        Alternatively, it can be a single file
        Mask can only take the value of either 0 or 1, and must have the shape
        of [height, width, depth, channel]
        `save_path`: save the csv of the results
        `quant_col`: attributes to be measured
        `quant_func`: function that quantifies such attributes; can only accept
        2 arguments: image and masks both as both numpy arrays
    '''
    if not os.path.isdir(mask_dir): mask = CTN.load_nib (mask_dir)
    if os.path.exists(save_path):
        ID = pd.read_csv (save_path)['ID']
        ID = [i+'.nii.gz' for i in ID]
    else: ID = []

    for i in os.listdir (img_dir):
        if not i in ID:
            img = CTN.load_nib (img_dir+'/'+i)
            if os.path.isdir (mask_dir):
                mask = CTN.load_nib (mask_dir +'/'+i)
            quant= quant_func (img, mask)
            quant_df = pd.DataFrame (quant.reshape([1,-1])) 
            quant_df.columns = quant_col
            quant_df ['ID'] = re.sub ('.nii.gz$', '', i)
            header = False if os.path.exists(save_path) else True
            quant_df.to_csv (save_path, mode='a', header=header)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--mask_dir', type=str)
    parser.add_argument('--template_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    CTN.makedir (args.save_dir)
    quant_col = choose_colnames (args.mode)
    if args.mode == 'tissue':
        sum_images (args.img_dir, args.mask_dir, 
                args.save_dir+'/tissue.csv', quant_col, BB.quant_tissue)
    elif args.mode =='artery_level1':
        sum_images (args.img_dir, args.template_dir+'/level1.nii', 
                args.save_dir+'/artery1.csv', quant_col, segment_territory)
    elif args.mode =='artery_level2':
        sum_images (args.img_dir, args.template_dir+'/level2.nii', 
                args.save_dir+'/artery2.csv', quant_col, segment_territory)
