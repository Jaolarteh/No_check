from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "test_auc": roc_auc_score(y_test, proba),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
