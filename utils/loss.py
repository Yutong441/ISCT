import numpy as np
import torch
import torch.nn.functional as F
import functools
import utils.metric as um
from typing import Optional

def MSELoss (output, target):
    if type (target) == list: target=target[0]
    return torch.mean ((output.squeeze ().to(torch.float) - 
        target.squeeze ().to(torch.float))**2)

def MAELoss (output, target):
    if type (target) == list: target=target[0]
    return torch.mean (torch.abs(output.squeeze ().to(torch.float) - 
        target.squeeze ().to(torch.float)))

def MSE_entropy (output, target, weighting=1):
    entropy = torch.nn.CrossEntropyLoss()
    return weighting*entropy (output, target) + MSELoss (
            torch.argmax (output, dim=1), target)

def BCE (ypred, ytrue, num_classes, logits=False):
    ytrue_hot = F.one_hot (ytrue, num_classes=num_classes)
    if logits: ypred = torch.sigmoid (ypred)
    ypred = ypred.clip (1e-10, 1- 1e-10)
    return -torch.mean (ytrue_hot*torch.log(ypred) + 
            (1-ytrue_hot)*torch.log (1-ypred))

def BCE_logits (ypred, ytrue, one_hot_code=True):
    if one_hot_code: ytrue = F.one_hot (ytrue)
    return -torch.mean (ytrue*ypred - torch.log (torch.exp (ypred) +1))

def ordinalMSE (output, target, num_class):
    target_ordinal = um.label2ordinal (target, num_class)
    return MSELoss (output, target_ordinal)

def ordinalBCE (output, target, num_class):
    ytrue = um.label2ordinal (target, num_class)
    return BCE_logits (output, ytrue, one_hot_code=False)

def KL_divergence (output, target, **kwargs):
    '''
    $\Sigma_x P(x)\log ( \frac {P(x)}{Q(x)} )$
    Args: `kwargs`: for `utils.label2gauss`
    Return: a torch scalar
    '''
    target_loggauss = um.label2gauss (target, **kwargs)
    norm_pred = torch.sum (torch.exp (output), dim=1, keepdim=True)
    ypred = torch.exp(output)/norm_pred
    return torch.mean (ypred*(output-torch.log (norm_pred)- target_loggauss))

def cumulative_link_loss (y_pred: torch.Tensor, y_true: torch.Tensor,
                         class_weights: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
    """
    from https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/losses.py
    Calculates the negative log likelihood using the logistic cumulative link
    function.
    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.
    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    Returns
    -------
    loss: torch.Tensor
    """
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true[:,None]), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    return neg_log_likelihood.mean()

def SVMLinear (output, target, num_class):
    target_ordinal = um.label2ordinal (target, num_class)
    return torch.mean ((1 - target_ordinal*output).clip(0)**2)

def unet_loss (output, target, cf):
    label_loss = get_loss_fun_basic (cf)
    return MSELoss (output[0], target[0]) + label_loss (output[1], target[1])

def hurdle_loss (output, target, thres):
    output, target = output.squeeze(), target.squeeze ()
    return MSELoss (output, target) + F.binary_cross_entropy (
            (output > thres).to (torch.float), 
            (target > thres).to (torch.float))

def two_stage_loss (output, target, thres):
    mse_out = MSELoss (output[0].squeeze(), target.squeeze())
    bce_out = F.binary_cross_entropy (output[1].squeeze(),
            (target > thres).squeeze().to(torch.float))
    return mse_out + 0.1*bce_out

def get_loss_fun_basic (cf):
    '''
    Choices of `cf['loss_type']`:
        classification: cross entropy
        regression: MSE
        hybrid: MSE + cross entropy
        ordinal_regression: MSE on expanded labels
        ordinal_classification: cross entropy on expanded labels
        ordinal_regression_coral: MSE on expanded labels with weight sharing
        ordinal_classification: cross entropy on expanded labels 
        ordinal_classification_coral: cross entropy on expanded labels with
        weight sharing
        KL: Kullback Leibler divergence
        cum_link: cumulative logistic link loss
        cum_link_BCE: BCE on the cumulative sigmoid function
        ordinal_SVM: loss function for linear SVM
        hurdle: MSE + binary cross entropy
        two_stage: a model outputs both regression and probability that the
        output is above a certain value (for zero-inflated distribution)
    '''
    if cf['loss_type'] == 'classification':
        return torch.nn.CrossEntropyLoss()
    elif cf['loss_type'] == 'regression': return MSELoss
    elif cf ['loss_type'] == 'MAE': return MAELoss
    elif cf['loss_type'] == 'hybrid': return MSE_entropy
    elif  'ordinal_regression' in cf['loss_type']: 
        return functools.partial (ordinalMSE, num_class=cf['predict_class'])
    elif  'ordinal_classification' in cf['loss_type']: 
        return functools.partial (ordinalBCE, num_class=cf['predict_class'])
    elif cf['loss_type'] == 'KL': 
        return functools.partial (KL_divergence, num_class=cf['predict_class'],
                device=cf['device'], sigma=cf['sigma'])
    elif cf['loss_type'] == 'cum_link': return cumulative_link_loss 
    elif cf['loss_type'] == 'cum_link_BCE': 
        return functools.partial (BCE, num_classes=cf['predict_class'])
    elif cf ['loss_type'] == 'ordinal_SVM': 
        return functools.partial (SVMLinear, num_class=cf['predict_class'])
    elif cf ['loss_type'] == 'SVM': return torch.nn.MultiMarginLoss()
    elif cf ['loss_type'] == 'hurdle': 
        return functools.partial (hurdle_loss, thres=cf['pos_label'])
    elif cf ['loss_type'] == 'two_stage':
        return functools.partial (two_stage_loss, thres=cf['pos_label'])

def get_loss_fun (cf):
    if 'unet' in cf ['model_type']: 
        return functools.partial (unet_loss, cf=cf)
    else: return get_loss_fun_basic (cf)

def match_param(param_list, model): 
    matched_param = []
    for i in param_list:
        for m, n in model.named_parameters ():
            if i in m: matched_param.append (m) 
    return matched_param

def add_L2 (cf, model):
    if cf['L2'] >0 and len (cf['regularised_layers']) >0:
        params_list = []
        matched_param = match_param (cf['regularised_layers'], model)
        if len (matched_param) >0:
            for i in matched_param:
                for m, n in model.named_parameters ():
                    if m in matched_param:
                        params_list.append (n.pow(2).sum())
            return cf['L2']*sum(params_list)
        else: return 0
    else: return 0

def add_L1 (cf, model):
    params_list = []
    if cf['L1'] >0 and len (cf['regularised_layers']) >0:
        matched_param = match_param (cf['regularised_layers'], model)
        if len (matched_param) >0:
            for i in matched_param:
                for m, n in model.named_parameters ():
                    if m in matched_param:
                        params_list.append (n.abs().sum())
            return cf['L1']*sum(params_list)
        else: return 0
    elif cf['L1'] >0 and len(cf['regularised_layers'])==0:
        for m, n in model.named_parameters ():
                params_list.append (n.abs().sum())
        return cf['L1']*sum(params_list)
    else: return 0
