from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]

print("TP/FN/FP/TN: count them yourself from the two lists")
print("accuracy :", accuracy_score(y_true, y_pred))
print("precision:", round(precision_score(y_true, y_pred), 3))
print("recall   :", recall_score(y_true, y_pred))
print("f1       :", round(f1_score(y_true, y_pred), 3))