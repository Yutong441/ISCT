'''
1. Combine the segmentation results
2. Create separate directories for registered and unregistered images
3. Save as compressed file
'''
import glob, argparse
import numpy as np
import data_raw.CTname as CTN

def combine_maps (ID, save_dir, key='wpull_'):
    img_dir = save_dir+'/'+ID
    mode = 'unregistered' if key == '' else 'registered'

    skullstripped = glob.glob (img_dir+'/'+key+'ss_*.nii')
    img = CTN.load_nib (skullstripped[0])
    CTN.makedir (save_dir+'/'+mode)
    CTN.save_nib (np.round (img), save_dir+'/'+mode+'/'+ID+'.nii.gz',
            dtype=np.int64)

    tissues = ['c01', 'c02', 'c03']
    all_maps = []
    for i in tissues:
        map_path = glob.glob (img_dir+'/'+key+i+'_*.nii')
        all_maps.append (CTN.load_nib (map_path[0]))
    all_maps = np.stack (all_maps, axis=-1)
    CTN.makedir (save_dir+'/'+mode+'_maps')
    CTN.save_nib (all_maps, save_dir+'/'+mode+'_maps/'+ID+'.nii.gz')

    masked = img[...,np.newaxis]*(all_maps>0.5).astype(float)
    CTN.makedir (save_dir+'/'+mode+'_masked')
    CTN.save_nib (masked, save_dir+'/'+mode+'_masked/'+ID+'.nii.gz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--index', type=str)
    args = parser.parse_args()
    for i in ['', 'wpull_']:
        combine_maps (args.index, args.save_dir, key=i)
