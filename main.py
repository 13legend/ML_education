import numpy as np

TP = [57, 48, 60, 55]
TN = [32, 35, 28, 41]
FP = [13, 15, 12, 11]
FN = [7, 12, 11, 8]

precision = np.mean([TP[i] / (TP[i] + FP[i]) for i in range(len(TP))])
recall = np.mean([TP[i] / (TP[i] + FN[i]) for i in range(len(TP))])
