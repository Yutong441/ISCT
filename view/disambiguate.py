# disambiguate a list of CT scans from a given patient
%matplotlib inline
import os
import IPython
os.chdir ('/PHShome/yc703/Documents/postICH')
import view.visual_select as VS
filename='data/original_labels/dicom_dup.csv'
save_path='data/original_labels/dicom_disamb.csv'
os.environ['PATH']=os.environ['PATH']+':/PHShome/yc703/.conda/envs/postICH/bin'

VS.multi_imshow_widget (df_path=filename, save_path= save_path,
        disp=IPython.display.display)
