from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch
from utils.spot import SPOT
from .affiliation import pr_from_events
from .affiliation.generics import convert_vector_to_events
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from .vus.metrics import get_range_vus_roc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def get_thres_by_SPOT(init_score, test_score, q=1e-2):
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold

def affiliation_SPOT(gt, anomaly_score, qs, thresholds, save_path, verbose=False):
    res_info = []
    for i, threshold in enumerate(thresholds):
        pred = (anomaly_score > threshold).astype(int)
        accuracy = accuracy_score(gt, pred)
        events_pred = convert_vector_to_events(pred)
        events_label = convert_vector_to_events(gt)
        Trange = (0, len(pred))
        result = pr_from_events(events_pred, events_label, Trange)
        P = result['precision']
        R = result['recall']
        F = 2 * P * R / (P + R)
        res = {
            'q': qs[i],
            'threshold': threshold,        
            'accuracy_affiliation': accuracy,
            'precision_affiliation': P,
            'recall_affiliation': R,
            'F1_affiliation': F,
        }
        if verbose: print(f"SPOT_{qs[i]}:\taffiliationF1_{F:.4f}")
        res_info.append(res)
    pd.DataFrame(res_info).to_csv(f"{save_path}/affiliation_SPOT.csv", index=False)
    return res_info

def F1_SPOT(gt, anomaly_score, qs, thresholds, save_path, verbose=False):
    res_info = []
    for i, threshold in enumerate(thresholds):
        pred = (anomaly_score > threshold).astype(int)
        accuracy = accuracy_score(gt, pred)
        precision, recall, F1, _ = precision_recall_fscore_support(gt, pred, average="binary")
        gt_PA, pred_PA = gt.copy(), pred.copy()  # copy
        gt_PA, pred_PA = _adjustment(gt_PA, pred_PA)
        precision_PA, recall_PA, F1_PA, _ = precision_recall_fscore_support(gt_PA, pred_PA, average="binary")
        res = {
            'q': qs[i],
            'threshold': threshold,        
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1': F1,
            'precision_PA': precision_PA,
            'recall_PA': recall_PA,
            'F1_PA': F1_PA,
        }
        if verbose: print(f"SPOT_{qs[i]}:\tF1{F1:.4f}\tF1PA_{F1_PA:.4f}")
        res_info.append(res)
    pd.DataFrame(res_info).to_csv(f"{save_path}/F1_SPOT.csv", index=False)
    return res_info

def affiliation(gt, anomaly_score, PARs, save_path, verbose=False):
    """

    Args:
        gt (np.array): 
        anomaly_score (np.array): 
        PARs (list): prior anomaly ratio, [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0]
        save_path(str): 
    """
    res_info = []
    for ratio in PARs:
        threshold = np.percentile(anomaly_score, 100 - ratio)
        pred = (anomaly_score > threshold).astype(int)
        accuracy = accuracy_score(gt, pred)
        events_pred = convert_vector_to_events(pred)
        events_label = convert_vector_to_events(gt)
        Trange = (0, len(pred))
        result = pr_from_events(events_pred, events_label, Trange)
        P = result['precision']
        R = result['recall']
        F = 2 * P * R / (P + R)
        res = {
            'ratio': ratio,
            'threshold': threshold,        
            'accuracy_affiliation': accuracy,
            'precision_affiliation': P,
            'recall_affiliation': R,
            'F1_affiliation': F,
        }
        if verbose: print(f"{ratio}:\taffiliationF1_{F:.4f}")
        res_info.append(res)
    pd.DataFrame(res_info).to_csv(f"{save_path}/affiliation.csv", index=False)
    return res_info

def auc_roc(gt, anomaly_score, save_path, verbose=False, vis=False):
    fpr, tpr, _ = roc_curve(gt, anomaly_score)
    auc_roc = auc(fpr, tpr)
    if vis:
        plt.plot(fpr, tpr, 'r--', label='ROC={0:.4f}'.format(auc_roc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f"{save_path}/auc_roc.png")
    if verbose: print(f"auc_roc:\t{auc_roc:.4f}")
    res = {"AUC_ROC": auc_roc}
    pd.DataFrame([res]).to_csv(f"{save_path}/auc_roc.csv", index=False)
    return auc_roc

def vus_roc(gt, anomaly_score, win_size, save_path, verbose=False):
    res = get_range_vus_roc(anomaly_score, gt, 100)
    pd.DataFrame([res]).to_csv(f"{save_path}/vus_roc.csv", index=False)
    # if verbose: print(f"vus_roc:\t{res['VUS_ROC']:.4f}")
    if verbose:
        print(f"vus_roc:\t{res['VUS_ROC']:.4f}")
        print(f"vus_pr:\t{res['VUS_PR']:.4f}")
    return res

def F1(gt, anomaly_score, PARs, save_path, verbose=False):
    res_info = []
    for ratio in PARs:
        threshold = np.percentile(anomaly_score, 100 - ratio)
        pred = (anomaly_score > threshold).astype(int)
        accuracy = accuracy_score(gt, pred)
        precision, recall, F1, _ = precision_recall_fscore_support(gt, pred, average="binary")
        gt_PA, pred_PA = gt.copy(), pred.copy()  # copy
        gt_PA, pred_PA = _adjustment(gt_PA, pred_PA)
        precision_PA, recall_PA, F1_PA, _ = precision_recall_fscore_support(gt_PA, pred_PA, average="binary")
        res = {
            'ratio': ratio,
            'threshold': threshold,        
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1': F1,
            'precision_PA': precision_PA,
            'recall_PA': recall_PA,
            'F1_PA': F1_PA,
        }
        if verbose: print(f"{ratio}:\tF1_{F1:.4f}\tF1PA_{F1_PA:.4f}")
        res_info.append(res)
    pd.DataFrame(res_info).to_csv(f"{save_path}/F1.csv", index=False)
    return res_info

def bestF1(lab, scores, save_path, verbose=False):
    scores = scores.numpy() if torch.is_tensor(scores) else scores
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones
    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    TPs = np.cumsum(-new_lab) + ones
    FPs = np.cumsum(new_lab-1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs
    N = len(lab) - np.flip(TPs > 0).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s  = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)
    res = {
            'precision': PPVs[maxid],
            'recall': TPRs[maxid],
            'F1': F1s[maxid],
        }   
    pd.DataFrame([res]).to_csv(f"{save_path}/best F1.csv", index=False)
    if verbose: print(f"best F1:\t{F1s[maxid]:.4f}")
    return res

def _adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def bestF1_label(y_true, y_preds, save_path, verbose=False):
    res = []
    for ratio in y_preds.keys():
        y_pred = y_preds[ratio]
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        res.append({
            'Ratio': ratio,
            'TP': int(tp),
            'TN': int(tn),
            'FN': int(fn),
            'FP': int(fp),
            'Accuracy': round(acc,3),
            'Precision': round(precision,3),
            'Recall': round(recall,3),
            'F1 Score': round(f1,3)
        })
    pd.DataFrame(res).to_csv(f"{save_path}/best F1.csv", index=False)
    if verbose: print(pd.DataFrame(res))
    return res

def evaluate(gt, anomaly_score, metrics, save_path, PARs=None, verbose=True):
    if PARs is None: PARs = [3.0, 5.0]
    if "auc_roc" in metrics: auc_roc(gt, anomaly_score, save_path, verbose) 
    if "vus_roc" in metrics: vus_roc(gt, anomaly_score, 100, save_path, verbose)
    if "best_f1" in metrics: bestF1(gt, anomaly_score, save_path, verbose)
    if "affiliation" in metrics: affiliation(gt, anomaly_score, PARs, save_path, verbose)
    if "f1" in metrics: F1(gt, anomaly_score, PARs, save_path, verbose)

def evaluate_SPOT(gt, init_score, anomaly_score, metrics, save_path, qs=None, verbose=True):
    if qs is None: qs = [0.001, 0.01, 0.1]
    thresholds = [get_thres_by_SPOT(init_score, anomaly_score, q) for q in qs]
    # thresholds = qs
    if verbose:
        print(qs)
        print(thresholds)
    if "auc_roc" in metrics: auc_roc(gt, anomaly_score, save_path, verbose) 
    if "vus_roc" in metrics: vus_roc(gt, anomaly_score, 100, save_path, verbose)
    if "best_f1" in metrics: bestF1(gt, anomaly_score, save_path, verbose)
    if "affiliation" in metrics: affiliation_SPOT(gt, anomaly_score, qs, thresholds, save_path, verbose)
    if "f1" in metrics: F1_SPOT(gt, anomaly_score, qs, thresholds, save_path, verbose)

def evaluate_tab(gt, anomaly_score, metrics, save_path, PARs=None, verbose=True, preds=None):
    if PARs is None: PARs = [3.0, 5.0]
    if "auc_roc" in metrics: auc_roc(gt, anomaly_score, save_path, verbose)
    if "vus_roc" in metrics: vus_roc(gt, anomaly_score, 100, save_path, verbose)
    if "best_f1" in metrics: bestF1_label(gt, preds, save_path, verbose)
    if "affiliation" in metrics: affiliation(gt, anomaly_score, PARs, save_path, verbose)
    if "f1" in metrics: F1(gt, anomaly_score, PARs, save_path, verbose)