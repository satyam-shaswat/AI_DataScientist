import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = [3, 5, 2, 7]
y_pred = [2, 5, 4, 8]

print("MAE :", mean_absolute_error(y_true, y_pred))
print("RMSE:", round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3))
print("R2  :", round(r2_score(y_true, y_pred), 3))