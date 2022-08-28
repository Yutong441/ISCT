import os
import re
import data_raw.CTname as CTN

def clean(x):
    return re.sub('.nii.gz', '', x)

def join_str(x, sep=','):
    return sep.join([str(i) for i in x])

def find_shapes(data_dir, save_dir):
    '''
    Obtain the shape of tensors stored as nii.gz in a directory.
    Save the results as a csv (`save_dir`).
    '''
    all_files = sorted(os.listdir(data_dir))
    shape_list = [clean(i)+','+join_str((CTN.load_nib(
        data_dir+'/'+i)).shape)+'\n' for i in all_files if 'nii.gz' in i]
    data_name = os.path.basename(re.sub('/$', '', data_dir))
    with open(save_dir+'/'+data_name+'_shape.csv', 'w') as f:
        f.writelines(shape_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    CTN.makedir(args.save_dir)
    find_shapes(args.data_dir, args.save_dir)
