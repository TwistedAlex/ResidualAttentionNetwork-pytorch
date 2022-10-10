from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as metrics


def save_roc_curve(labels, predictions, epoch_num, path):

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, auc, threshold = roc_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(path, "roc_curve_{}".format(epoch_num) + '_'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"))
    plt.cla()


def save_roc_curve_with_threshold(labels, predictions, epoch_num, path, fpr_threshold = 0.1):
    np.save(path+"/labels", labels)
    np.save(path+"/predictions", predictions)
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, auc, threshold = roc_curve(labels, predictions)
    index_fpr_threshold = 0

    # max_xy = max(fpr[-index_fpr_threshold], tpr[-index_fpr_threshold])

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 0.1], [0.9, 1], 'r--')
    plt.xlim([0, 0.1])
    plt.ylim([0.9, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(path, "roc_curve_with_threshold_{}".format(epoch_num) + '_'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"))
    plt.cla()


def calc_sensitivity(labels, predictions, false_alarm_points=None, return_thresholds_for_fa=False):
    if false_alarm_points is None:
        false_alarm_points = [.1 / 100, .05 / 100, .3 / 100, .5 / 100]

    fpr, tpr, auc, thresholds = roc_curve(labels, predictions)
    all_sens = []
    all_thresholds = []
    for fa in false_alarm_points:
        # output the false positive value when true positive rate = false_alarm_point
        sensitivity_at_fa = np.interp(x=fa, xp=fpr, fp=tpr)
        # output the true positive value when false positive rate = false_alarm_point
        th_at_fa = np.interp(x=fa, xp=fpr, fp=thresholds)
        all_sens += [sensitivity_at_fa]
        all_thresholds += [th_at_fa]

    if return_thresholds_for_fa:
        return all_sens, all_thresholds
    return all_sens, auc, fpr, tpr


def roc_curve(labels, preds, thresholds_count=10000):
    if len(labels) == 1:
        raise Exception(f'roc_curve: labels parameter is empty')
    if len(np.unique(labels)) == 1:
        raise Exception(f'roc_curve: labels parameter is composed of only one value')

    preds_on_positive = preds[labels == 1]
    preds_on_negative = preds[labels == 0]
    min_negative = min(preds_on_negative)
    max_positive = max(preds_on_positive)
    margin = 0  # (max_positive - min_negative)/100

    thresholds = np.linspace(min_negative - margin, max_positive + margin, thresholds_count)
    true_positive_rate = [np.mean(preds_on_positive > t) for t in thresholds]
    spec = [np.mean(preds_on_negative <= t) for t in thresholds]
    false_positive_rate = [1 - s for s in spec]
    auc = np.trapz(true_positive_rate, spec)

    thresholds = np.flip(thresholds, axis=0)
    false_positive_rate.reverse(), true_positive_rate.reverse()
    false_positive_rate, true_positive_rate = np.asarray(false_positive_rate), np.asarray(true_positive_rate)
    return false_positive_rate, true_positive_rate, auc, thresholds
