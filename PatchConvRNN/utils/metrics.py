import numpy as np
from sklearn.metrics import r2_score, max_error, median_absolute_error, explained_variance_score
import torch


def RSE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true, threshold=-1e9):
    
    mask = np.logical_and(true > threshold, pred > threshold)

    
    pred_threshold = pred[mask]
    true_threshold = true[mask]

    
    if pred_threshold.size > 0 and true_threshold.size > 0:
        
        u = ((true_threshold - true_threshold.mean(axis=-1, keepdims=True)) *
             (pred_threshold - pred_threshold.mean(axis=-1, keepdims=True))).sum(axis=-1)

        
        d = np.sqrt(
            ((true_threshold - true_threshold.mean(axis=-1, keepdims=True)) ** 2 *
             (pred_threshold - pred_threshold.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1))
        d += 1e-12  

        
        return 0.018 * (u / d).mean()
    else:
        
        return None


def MAE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.abs(pred - true))


def MSE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.mean((pred - true) ** 2)


def RMSE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.square((pred - true) / true))


def MAX_error(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return max_error(true, pred)


def Median_Absolute_Error(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return median_absolute_error(true, pred)


def EV_scores(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return explained_variance_score(true, pred)


def mean_std(pred, true, threshold=-1e9,group_size=None):
    if isinstance(pred, torch.Tensor):
        pred, true = pred.numpy(), true.numpy()
    if pred.ndim > 1 and true.ndim > 1:
        pred = pred.reshape(-1)
        true = true.reshape(-1)
    if group_size is None:
        raise ValueError("group_size must be specified")

    mask = (true > threshold) & (pred > threshold)
    pred, true = pred[mask], true[mask]
    stds_pred, stds_true = [], []

    for i in range(0, len(pred), group_size):
        stds_pred.append(np.std(pred[i:i + group_size],ddof=1))

    for i in range(0, len(true), group_size):
        stds_true.append(np.std(true[i:i + group_size],ddof=1))
    mean1 = np.mean(stds_pred)
    mean2 = np.mean(stds_true)
    return np.abs(mean1 - mean2)


def R2_scores(pred, true, threshold=-1e9):
    mask = np.logical_and(true > threshold, pred > threshold)
    pred = pred[mask]
    true = true[mask]
    return r2_score(true, pred)


def adjust_R2_scores(R2_score, num_data, num_feature):
    return 1 - (1 - R2_score) * (num_data - 1) / (num_data - num_feature - 1)


def metric(pred, true, num_data=None, num_feature=None,pred_len=None,flag=None):
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)
        rse = RSE(pred, true)
        corr = CORR(pred, true)
        Max_error = MAX_error(pred, true)
        Median_absolute_error = Median_Absolute_Error(pred, true)
        ev_scores = EV_scores(pred, true)
        r2_scores = R2_scores(pred, true)
        Adjust_R2_scores = adjust_R2_scores(r2_scores, num_data, num_feature)
        STD = mean_std(pred, true, group_size=pred_len)
        return mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores, r2_scores, Adjust_R2_scores,STD

