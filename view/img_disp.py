%matplotlib inline
import os
os.chdir ('/PHShome/yc703/Documents/postICH/')
import view.visual_select as VS
import data_raw.CTname as CTN
df_path = 'data/original_labels/dicom_path.csv'

imgID = 'MGH062'
img = DTN.load_nib ('data/tmp_images/'+imgID+'.nii.gz')
print (img.shape)
VS.show_levels (img)

VS.show_ori_dcm (imgID, df_path)
