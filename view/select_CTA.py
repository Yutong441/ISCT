%matplotlib inline
import os
import pandas as pd
os.chdir ('/PHShome/yc703/Documents/postICH/')
import view.visual_select as VS
import data_raw.CTname as CTN
root_dir='/data/rosanderson/is_ct/processed/registered/'

imgID = pd.read_csv ('data/tmp_images/missing_reg_trim.txt', header=None)
img = CTN.load_nib (root_dir+imgID[0].iloc[0])
print (imgID[0].iloc[0])
print (img.max())
VS.show_levels (img)

