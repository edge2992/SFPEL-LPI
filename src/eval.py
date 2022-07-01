import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt


def plot_ROC(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label="ROC curve (area= %0.2f)" % roc_auc,
    )
    plt.legend()
    plt.show()


def evaluation_metric(y_ture, y_score):
    # f1 scoreが一番高いthresholdを選び、評価をする
    precisions, recalls, thresholds = precision_recall_curve(y_ture, y_score)
    fscore = (2 * precisions * recalls) / (precisions + recalls + 1e-20)

    ix = np.argmax(fscore)
    y_pred = y_score >= thresholds[ix]
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred).ravel()

    accuracy = accuracy_score(y_ture, y_pred)
    precision = precisions[ix]
    f1 = fscore[ix]
    sen = recalls[ix]
    spec = tn / (tn + fp)

    print(
        "sensitivity: {:.3f}, specificity: {:.3f}, precision: {:.3f}, accuracy: {:.3f}, f1: {:.3f}".format(
            sen, spec, precision, accuracy, f1
        )
    )
    return (sen, spec, precision, accuracy, f1)
