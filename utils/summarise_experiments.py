import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import data_raw.CTname as CTN

def sum_one_experiment (folder, metrics=['AUC', 'MSE']):
    summary = {}
    summary ['name'] = re.sub('_\d{14}', '', CTN.get_basename (folder))
    summary ['time'] = CTN.find_datetime (folder)
    acc_folder = glob.glob (folder+'/*tmp/')[0]
    acc = pd.read_csv (acc_folder+'/acc_metric.csv', index_col=[0])
    for metric in metrics:
        for i in range(len(acc)):
            key = metric+'_'+re.sub (' \(.*\)$', '', acc.index[i])
            summary [key] = acc [metric].iloc[i]

    train = pd.read_csv (glob.glob(folder+'/*metric_test.csv')[0])
    summary ['epoch'] = len (train)
    epochs = train[metrics[0]].values
    summary ['best_epoch'] = np.where(epochs == max(epochs))[0][0]

    json_file = glob.glob (folder+'/*.json')[0]
    with open (json_file, 'r') as f: config = json.load (f)
    comb = {**summary, **config}
    comb = pd.DataFrame.from_dict (comb, orient='index')
    return comb.T

def sum_all_experiments (root):
    all_exp = [sum_one_experiment (root+'/'+i
        ) for i in os.listdir (root) if os.path.isdir(root+'/'+i)]
    all_exp = pd.concat (all_exp, axis=0, ignore_index=True)
    all_exp = all_exp.sort_values (by=['time'])
    all_exp.to_csv (root+'/summary.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    args = parser.parse_args()
    sum_all_experiments (args.log_dir)
