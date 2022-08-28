import argparse
import numpy as np
import data_raw.CTname as CTN
import nibabel as nib

def label2img (img_path, save_path, background=0):
    '''convert template with labels to binary mask image'''
    nibimg = nib.load (img_path)
    img = np.squeeze (nibimg.get_fdata())
    uniq_vec = np.sort (np.unique (img))
    uniq_vec = uniq_vec [uniq_vec!=background] #exclude background
    bin_img = img[...,np.newaxis] == uniq_vec.reshape([1,1,1,-1])
    CTN.makedir (save_path)
    for i in range (bin_img.shape[-1]):
        save_nib = nib.Nifti1Image (bin_img[...,i].astype(float), 
                affine=nibimg.affine)
        save_nib.to_filename(save_path+'/label'+str(i)+'.nii')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    label2img (args.img_path, args.save_path)
