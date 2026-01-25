import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(tp, tn, fp, fn, title="Confusion Matrix"):
    """
    Plotta una confusion matrix a partire dai valori TP, TN, FP, FN.

    Args:
        tp (int): True Positive
        tn (int): True Negative
        fp (int): False Positive
        fn (int): False Negative
        title (str): Titolo del plot
    """
    # Creiamo la matrice 2x2
    cm = np.array([[tp, fn],
                   [fp, tn]])

    # Creiamo il plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, xticklabels=['Positive','Negative'], yticklabels=['Positive','Negative'])
    plt.title(title, fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()


TP = 3
FP = 1
TN = 11
FN = 3

plot_confusion_matrix(tp=TP, tn=TN, fp=FP, fn=FN, title="Confusion Matrix - ladderActivity (Threshold 0.7)")
