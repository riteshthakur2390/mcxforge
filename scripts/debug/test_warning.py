import warnings
from sklearn.exceptions import FitFailedWarning

warnings.simplefilter('always')
from ml.model import SignalForgeEnsemble
import numpy as np
import pandas as pd

X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
y = pd.Series(np.random.randint(0, 2, 100))

model = SignalForgeEnsemble()
model.build()
model.evaluate_cv(X, y, [f'f{i}' for i in range(10)])
