import os, re
import numpy as np
import pandas as pd
import data_raw.CTname as CTN
import skimage

def empty_index (vec):
    indices = np.where (vec>1)[0]
    return min(indices), max(indices)

def bbox (img):
    if len(img.shape)==4: img = img.sum(3)
    empty_x = empty_index (img.sum(2).sum(1))
    empty_y = empty_index (img.sum(2).sum(0))
    empty_z = empty_index (img.sum(1).sum(0))
    return np.array (empty_x + empty_y + empty_z)

def common_bbox (img_dir):
    indices, files= [], []
    for img_path in os.listdir (img_dir):
        img = CTN.load_nib (img_dir+'/'+img_path)
        img = normalize (img)
        try: 
            indices.append (bbox (img))
            files.append (img_path)
        except: print ('cannot find bounding box for '+img_path)
    indices = np.transpose (np.stack (indices, axis=-1))
    df = pd.DataFrame (indices)
    df.columns = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
    df.index = files
    df.to_csv (os.path.dirname (img_dir)+'/bbox.csv')
    common = np.round (indices.mean(axis=0))
    return common.astype(int)

def apply_bbox (img, bindex):
    return img [bindex[0]:bindex[1], bindex[2]:bindex[3], bindex[4]:bindex[5]]

def normalize (img, window=[0,100]):
    img = img.clip (window[0], window[1])
    if len (img.shape) == 3: return img/window[1]
    elif len (img.shape) == 4:
        img -= img.min (axis=-1, keepdims=True)
        return img/img.max (axis=-1, keepdims=True)

def remove_blank (img, thres=0.1):
    '''
    Args:
        `thres`: percentage of non-zero pixel below which a particular slice is
        removed
    '''
    if len (img.shape)==3: inp=img
    elif len (img.shape)==4: inp=img.sum(-1)
    H, W = img.shape[:2]
    z_perc = (inp>1).sum(axis=(0,1))/(H*W)
    return img [:,:,z_perc>thres]

def sel_interval (img, dcm_thick=1000, min_thick=5):
    '''
    Args:
        `dcm_path`: path to the folder of nifty files
        `dcm_thick`: thickness of each slice in `dcm_path`
        `min_thick`: minimum thickness of each slice in mm
    Return:
        a 3D numpy array of shape [H, W, D] 
    '''
    N = img.shape[2]
    if dcm_thick == 1000:
        if N<50: dcm_thick = 5
        elif N>=50 and N<100: dcm_thick = 2.5
        elif N>=100 and N<250: dcm_thick = 1
        elif N>=250: dcm_thick=0.5

    interval = int(np.round(min_thick/dcm_thick))
    return img [...,::interval]

def record_missing (ID, save_dir):
    save_path = os.path.dirname (re.sub ('/$', '', save_dir))
    base = CTN.get_basename (save_dir)
    with open (save_path+'/missing_'+base+'.txt', 'a') as f: 
        f.write(ID+'\n')

def trim_img (img_dir, save_dir, bindex=None, to_shape=None, CTA_thres=120,
        bone_thres=1000):
    '''
    Method:
    1. remove excess background pixels (optional)
    2. normalize the image between 0 and 1
    3. remove slices that do not contain sufficient brain tissues
    4. downsample z axis to 5mm thick
    5. downsample image (optional)
    '''
    CTN.makedir (save_dir)
    for img_path in os.listdir (img_dir):
        img = CTN.load_nib (img_dir+'/'+img_path)
        if bindex is not None: img = apply_bbox (img, bindex)
        img = remove_blank (img)
        if img.shape[2]==0: 
            record_missing (img_path, save_dir); continue

        # flag images that could potentially be  CTA
        if img.max()>CTA_thres: record_missing (img_path, save_dir)
        # remove images that still contain bone tissues
        if img.max()>bone_thres: 
            record_missing (img_path, save_dir); continue

        img = normalize (img)
        img = sel_interval (img) 
        if to_shape is not None: 
            img = skimage.transform.resize (img, to_shape, order=1)
        CTN.save_nib (img, save_dir+'/'+img_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--shape', type=str, default='None')
    parser.add_argument('--bindex', type=str, default='None')
    parser.add_argument('--CTA_thres', type=str, default='120')
    args = parser.parse_args()

    if args.shape != 'None': 
        to_shape= [int(i) for i in args.shape.split(',')]
    else: to_shape=None
    if args.bindex != 'None':
        #bindex = common_bbox (args.img_dir)
        bindex= np.array ([24,154,34,204,77,225])
    else: bindex=None
    trim_img (args.img_dir, args.save_dir, bindex, to_shape,
            float(args.CTA_thres))
