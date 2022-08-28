import os, argparse
import numpy as np
import data_raw.CTname as CTN

def img2label (img_dir, save_path):
    imgs= []
    for i in sorted (os.listdir (img_dir)):
        one_img = CTN.load_nib (img_dir+'/'+i)
        imgs.append (one_img>0.5)
    label = np.stack (imgs, axis=-1)
    CTN.save_nib (label.astype(int), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    img2label (args.img_dir, args.save_path)
