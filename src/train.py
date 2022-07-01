from src.loader import InputData
from src.model import LPI


data = InputData()
Y = data.InteractionMatrix
X = [data.PCPseDNCFeature_LNCRNA, data.SCPseDNCFeature_LNCRNA]
model = LPI()
model.fit(X, Y)
