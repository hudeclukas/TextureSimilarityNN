import numpy as np
from matplotlib import pyplot as plt

def tpr_fpr(distances, y_true, threshold):
    tp = sum(1 for d,y in zip(distances,y_true) if d < threshold and y == 1)
    fp = sum(1 for d,y in zip(distances,y_true) if d < threshold and y == 0)
    tn = sum(1 for d,y in zip(distances,y_true) if d > threshold and y == 0)
    fn = sum(1 for d,y in zip(distances,y_true) if d > threshold and y == 1)
    tcond = tp + fn
    if tcond > 0:
        tpr = tp / tcond
    else:
        tpr = 0
    fcond = tn + fp
    if fcond > 0:
        fpr = fp / fcond
    else:
        fpr = 0
    return tpr, fpr

def plot_ROC(tprs, fprs, thresholds, label, color, save_path=None) -> [float,float]:
    AUC = np.trapz(tprs, fprs)
    idx = np.argmax(np.array(tprs) - np.array(fprs))
    t = thresholds[idx]
    plt.figure(figsize=[12,8])
    plt.plot(fprs, tprs, color=color, label=label+f' AUC={AUC:.3f}')
    plt.plot([1, 0], [1, 0], color='navy', linestyle='--', label='50% random probability')
    plt.plot([fprs[idx], fprs[idx]], [0, tprs[idx]], color='black', linestyle=':',
             label='fpr:{:0.3f}, tpr:{:0.3f}, thr:{:0.3f}'.format(fprs[idx], tprs[idx], thresholds[idx]))
    plt.plot([0, fprs[idx]], [tprs[idx], tprs[idx]], color='black', linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if save_path != None:
        plt.savefig(save_path, dpi=200.0)
    plt.show()
    plt.close()
    return t, AUC