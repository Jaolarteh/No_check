import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold,
    mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class HybridFeatureSelector:
    """
    Combina múltiples técnicas de selección:
    1. Variance Threshold
    2. Importancia con modelo lineal L1
    3. Importancia con XGBoost
    4. Mutual Information
    """

    def __init__(self,
                 var_threshold=0.0,
                 use_l1=True,
                 use_xgb=True,
                 use_mutual_info=True,
                 top_k=30):
        self.var_threshold = var_threshold
        self.use_l1 = use_l1
        self.use_xgb = use_xgb
        self.use_mutual_info = use_mutual_info
        self.top_k = top_k
        self.selected_features_ = None

    def fit(self, X, y):
        scores = {}

        # 1. Variance threshold
        if self.var_threshold > 0:
            vt = VarianceThreshold(self.var_threshold)
            vt.fit(X)
            scores["variance"] = vt.variances_

        # 2. Logistic Regression L1
        if self.use_l1:
            lr = LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=200
            )
            lr.fit(X, y)
            scores["l1"] = np.abs(lr.coef_[0])

        # 3. XGBoost Importance
        if self.use_xgb:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss"
            )
            xgb.fit(X, y)
            scores["xgb"] = xgb.feature_importances_

        # 4. Mutual Information
        if self.use_mutual_info:
            scores["mi"] = mutual_info_classif(X, y, random_state=42)

        # --- Normalización de puntajes ---
        df_scores = pd.DataFrame(scores, index=X.columns)
        df_scores = df_scores.fillna(0)
        df_scores["score_total"] = df_scores.sum(axis=1)

        # --- Selección final ---
        selected = df_scores.sort_values("score_total", ascending=False)
        self.selected_features_ = selected.head(self.top_k).index.tolist()

        return self

    def transform(self, X):
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
