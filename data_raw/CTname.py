import os, re, glob, argparse, subprocess, json, shutil, datetime
import numpy as np
import pandas as pd
import nibabel as nib

# ====================File management====================
def get_basename (xx):
    return os.path.basename (re.sub ('/$', '', xx))

def bash_in_python (bashcommand):
    '''
    Run bash command inside python by writing the command into a script and
    executing that script. This method, however, tends to precipitate core dump
    error when running matlab. Thus, it is best to run matlab command from a
    bash script directly.
    '''
    ID=''.join ([str(i) for i in np.random.choice(9,10)])
    sh_script = 'tmp_test'+ID+'.sh'
    with open (sh_script, 'w') as f:
        f.write ('#!/bin/bash \n')
        f.write (bashcommand)
    subprocess.run (['chmod', '+x', sh_script], stderr=subprocess.PIPE)
    subprocess.call ('./'+sh_script)
    os.remove (sh_script)

def makedir (directory):
    if not os.path.exists (directory): os.mkdir (directory)

def load_nib (img_path):
    img = nib.load (img_path)
    return img.get_fdata ()

def save_nib (img, img_path, dtype=None):
    img_nib = nib.Nifti1Image (img, None, dtype=dtype)
    img_nib.to_filename(img_path)

def get_depth (img_path, to_shape=[128,128]):
    try:
        img = load_nib (img_path)
        return img.shape[-1]
    except: return 0

# ====================Dicom tag info====================
def find_datetime (filename): 
    '''If in dcm2niix, the `-f` argument contains %t, the scan time will be
    printed as a 14-digit string in the order of year-month-day-hour-
    minute-second. This function extracts that 14-digit string and convert it
    back to datetime format
    '''
    date_of_scan = re.search('.*(\d{14}).*', filename).group(1)
    # datetime does not count leap seconds
    if re.search ('60$', date_of_scan):
        date_of_scan = re.sub ('60$', '00', date_of_scan)
        return datetime.datetime.strptime(date_of_scan, '%Y%m%d%H%M%S') + \
                datetime.timedelta(minutes=1)
    else: return datetime.datetime.strptime(date_of_scan, '%Y%m%d%H%M%S')

def find_json_file (filename):
    options = glob.glob (filename)
    if len(options) == 0:
        options = glob.glob (re.sub ('__', '_', filename))
    return options

def remove_tags ():
    ''' Remove certain dicom tags are not useful for the project '''
    return ['InstitutionName', 'InstitutionalDepartmentName',
            'ConvolutionKernel', 'ExposureTime', 'XRayTub', 'eCurrent',
            'XRayExposure', 'ImageOrientationPatientDICOM',
            'ConversionSoftwareVersion']

def extract_tags (dirs, view):
    '''
    Extracting dicom tags of a scan series using dcm2niix
    Args:
        `dirs`: a list containing the directory path of dicom files, and
        the path to save the tags
        `view`: specify which series in `dirs` to extract dicom tags
    Return:
        a pandas dataframe containing the dicom tags
    '''
    bash_in_python ('dcm2niix -z y -v n -f "{}"_%t_%s -o {} {}'.format (
        view, dirs[1], dirs[0]+'/'+view))
    # `-f "{}_%t"` return images with scan date as a part of the name
    # `-v n` succinct output (many images to be processed)
    # `-o {}` output directory

    options = find_json_file (dirs[1]+'/'+view+'*.json')
    if len(options) >0:
        json_list = []
        for i in options:
            with open (i, 'r') as f: json_data =json.load (f)
            json_data ['scan_date'] = find_datetime (i)
            json_data ['num'] = get_depth(re.sub('.json', '.nii.gz', i))
            json_data ['view'] = view
            json_data ['out_file'] = os.path.basename (i)
            json_data =pd.DataFrame.from_dict (json_data, orient='index')
            json_list.append (json_data)
        json_data = pd.concat (json_list, axis=1)
    else: 
        json_data = {'scan_date': 'no data'}
        json_data =pd.DataFrame.from_dict (json_data, orient='index')
        json_data ['view']= view
        json_data ['num'] = 0
    # sometimes the 'terminal' directory may not contain any dicom images
    # thus dcm2niix may not return a json file
    return json_data.T

# ====================Loop over scans and patients====================
def views_per_mode (dirs, mode):
    dirs = [i+'/'+mode for i in dirs]
    makedir (dirs[1])
    all_views = sorted (os.listdir (dirs[0]))
    tags = [extract_tags (dirs, i) for i in all_views]
    tags = pd.concat (tags, axis=0, ignore_index=True)
    # need `ignore_index=True` in order to merge overlapping columns

    view_dict = {i:[len(os.listdir(dirs[0]+'/'+i))]
            for i in all_views }
    view_df = pd.DataFrame.from_dict(view_dict, orient='index')
    view_df ['mode'] = get_basename (dirs[0])
    view_df.index = np.arange(len(view_df))

    accession = os.path.dirname (re.sub ('/$', '', dirs[0]))
    view_df ['accession'] = get_basename (accession)
    view_df ['patientID'] = get_basename (os.path.dirname (accession))
    out_df = pd.concat ([view_df, tags], axis=1)
    out_df = out_df.drop (remove_tags (), axis=1, errors='ignore')
    shutil.rmtree(dirs[1])
    return out_df

def modes_per_scan (dirs, accession):
    dirs = [i+'/'+accession for i in dirs]
    makedir (dirs[1])
    mode_df = [views_per_mode (dirs,i) for i in os.listdir ( dirs[0])]
    shutil.rmtree(dirs[1])
    if len (mode_df) > 1: 
        return pd.concat (mode_df, axis=0, ignore_index=True)
    elif len (mode_df) == 1: return mode_df [0]

def scans_per_subject (dirs, patient):
    '''Args:
        `pattient`: directionary containing all scans from one patient
    '''
    dirs = [i+'/'+patient for i in dirs]
    makedir (dirs[1])
    all_patients = os.listdir (dirs[0])
    if len (all_patients) >0:
        scan_df = [modes_per_scan (dirs,i) for i in all_patients]
        patient_df = pd.concat (scan_df, axis=0)
    else:
        null_dict = {'num': 0, 'view': 'NA', 'accession': 'NA', 
                'patientID': get_basename (dirs[0])}
        patient_df = pd.DataFrame (null_dict.items())
    patient_df.to_csv (dirs[1]+'/'+'metadata.csv')

def comb_csv (csv_dir):
    csv_list = [pd.read_csv(csv_dir+'/'+i+'/metadata.csv'
        ) for i in os.listdir (csv_dir) if os.path.isdir(csv_dir+'/'+i)]
    return pd.concat (csv_list, axis=0, ignore_index=True)

def filenum (dirname, save_dir, log_dir, start=0, end=-1):
    '''
    Obtain information for each dicom folder, including patient ID, scan time,
    slice number etc. Depending on the number of scans each patient has, this
    function may take 30 seconds to 5 minutes per patient. It can take a long
    time for a large sample.
    Args:
        `dirname`: name of the directory containing images of different
        patients, assuming the structure is:
        `save_dir`: where to save the output

    dirname
    ----patient1
    --------scan1
    ------------CT
    ----------------view1
    --------------------0001.dcm
    --------------------0002.dcm
    --------------------0003.dcm
    ----------------view2
    --------------------0001.dcm
    --------------------0002.dcm
    --------------------0003.dcm
    --------scan2
    ----patient2

    Returns:
        save a pandas dataframe into a csv file
    '''
    makedir (save_dir)
    makedir (log_dir)
    if end==-1: end=len (os.listdir (dirname))
    log_file= save_dir+'/log'+str(end)
    if os.path.exists(log_file): os.remove (log_file)
    for subject in os.listdir (dirname)[start:end]:
        if os.path.isdir (dirname+'/'+subject):
            with open (log_file, 'a') as f: f.write (subject+'\n')
            scans_per_subject ([dirname, save_dir], subject) 
    #scans_per_subject ([dirname, save_dir], '0002585')
    all_df = comb_csv (save_dir)
    all_df.to_csv (log_dir+'/'+get_basename (dirname)+'.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()
    filenum (args.scan_dir, args.save_dir, args.log_dir, args.start,
            args.end)
