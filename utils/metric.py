import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics

# categorical classification
def reverse_one_hot (arr: np.ndarray):
    arr = np.squeeze (arr)
    if len (arr.shape) == 2 and arr.shape[1] != 2:
        return np.argmax (arr, axis=1)
    elif len (arr.shape) == 2 and arr.shape[1] == 2:
        return arr [:,1]
    else: return arr

# ordinal classification
def ordinal2label(pred: np.ndarray):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.1, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.1, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.1, 0.1] -> 2
    [0.9, 0.9, 0.9, 0.1] -> 3
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) 

def label2ordinal (label: torch.tensor, num_class: int):
    assert max (label) <= num_class - 1, \
            'num_class should be larger than the maximum label'
    flip_hot =torch.flip (F.one_hot (label, num_classes=num_class), dims=[1])
    return torch.flip (torch.cumsum (flip_hot, dim=1), dims=[1])[:,1:]

def label2gauss (label: torch.tensor, num_class: int, sigma=0.1, 
        device: str='cpu'):
    num = torch.arange(0, num_class, device=device)
    if type (sigma) == np.ndarray: 
        assert len (sigma) == num_class, \
                'the length of sigma should equal to number of classes'
        sigma = torch.tensor (sigma, device=device)[label][:,None]
    expo = -(num [None]- label[:,None])**2/(2*sigma**2)
    norm_label = torch.log (torch.sum (torch.exp (expo), dim=1, keepdim=True) )
    return expo - norm_label

def AUC_MSE (output: np.ndarray, target: np.ndarray,
        loss_type='classification', pos_label=2):
    if 'ordinal' not in loss_type: output_bi = reverse_one_hot (output)
    else: output_bi = ordinal2label (output)
    if loss_type == 'hurdle_loss':
        output_bi *= (output_bi>pos_label).astype(float)

    fpr, tpr, _ = metrics.roc_curve (target>pos_label, output_bi)
    acc = np.mean (np.round (output_bi) == target)
    tn, fp, fn, tp= metrics.confusion_matrix (target>pos_label, 
            output_bi>pos_label, labels= [0,1]).ravel()
    return np.array ([metrics.auc (fpr, tpr), np.mean ((output_bi- target)**2),
        acc, tp/(tp+fn), tn/(tn+fp)])

def model_accuracy (result_dir, label_dir, ycol='mRS_new'):
    ypred = np.load (result_dir+'/CTP_tmp/ypred_test.npz')['arr_0']
    ytrue = pd.read_csv (label_dir+'/test_y.csv')[ycol]
    ypred_bool, ytrue_bool = ypred > 2, ytrue >2
    print ('accuracy: {}'.format (np.mean (ypred_bool == ytrue_bool)))

    fpr, tpr, _ = metrics.roc_curve (ypred_bool, ytrue_bool)
    tn, fp, fn, tp= metrics.confusion_matrix (ytrue_bool, ypred_bool, labels=
            [0,1]).ravel()
    print ('AUC: {}'.format (metrics.auc (fpr, tpr)))
    print ('sensitivity: {}'.format (tp/(tp+fn))) 
    print ('specificity: {}'.format (tn/(tn+fp)))

def rescale_mRS (xx):
    xx [xx >= 3.75] = 6.
    xx [np.logical_and (xx >= 3.25, xx < 3.75)] = 5.
    xx [np.logical_and (xx >= 2.75, xx < 3.25)] = 4.
    xx [np.logical_and (xx >= 2.25, xx < 2.75)] = 3.
    xx [np.logical_and (xx >= 1.5, xx < 2.25)] = 2.
    return np.round (xx, 0).astype (int)
