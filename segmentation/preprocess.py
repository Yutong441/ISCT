import re, os, subprocess, glob
import numpy as np
import pandas as pd
import data_raw.CTname as CTN

def preprocess (dcm_paths, save_dir, index):
    dcm_df = pd.read_csv (dcm_paths)
    i= np.where (dcm_df ['anoID'].values == index)[0][0]
    ID = dcm_df['anoID'].iloc[i]
    save_ID_dir = save_dir+'/'+ID
    CTN.makedir (save_ID_dir)
    img_dir = dcm_df['directory'].iloc[i]
    dcm2niix_out = save_ID_dir+'/'+re.sub ('.json$', '.nii', 
            dcm_df['out_file'].iloc[i])
    view_name = dcm_df['view'].iloc[i]

    CTN.bash_in_python ('dcm2niix -v n -f "{}"_%t_%s -o {} {}'.format (
        view_name, save_ID_dir, img_dir))
    os.rename(dcm2niix_out, save_ID_dir+'/'+ID+'.nii')
    num = dcm_df['num'].iloc[i]
    if num >100:
          img = CTN.load_nib (save_ID_dir+'/'+ID+'.nii')
          if num > 200: img = img[...,::5]
          elif num > 100 & num <= 200: img = img [...,::2]
          CTN.save_nib (img, save_ID_dir+'/'+ID+'.nii')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_paths', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--index', type=str)
    args = parser.parse_args()
    preprocess (args.dcm_paths, args.save_dir, args.index)
