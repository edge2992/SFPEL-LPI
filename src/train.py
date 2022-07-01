from sklearn.metrics import auc, roc_curve
import numpy as np
from src.loader import InputData
from src.model import LPI


data = InputData()
Y = data.InteractionMatrix
X = [data.PCPseDNCFeature_LNCRNA, data.SCPseDNCFeature_LNCRNA]
model = LPI(max_iter=3)
model.fit(X, Y)
y_pred = model.predict(X, check=False)
fpr, tpr, _ = roc_curve(np.ravel(Y), np.ravel(y_pred))
print("auc: {:.3f}".format(auc(fpr, tpr)))
