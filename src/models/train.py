import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.data.load_data import load_raw_data
from src.features.preprocessing import build_preprocessor
from src.features.feature_selection import HybridFeatureSelector
from src.evaluation.metrics import evaluate_model
from src.utils.logging import get_logger

def build_models(preprocessor, feature_names, selector_top_k=30):
    to_dataframe = FunctionTransformer(
        lambda X: pd.DataFrame(X, columns=feature_names),
        feature_names_out=lambda _: feature_names,
    )

    def build_pipeline(model):
        selector = HybridFeatureSelector(top_k=selector_top_k)
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("to_dataframe", to_dataframe),
                ("selector", selector),
                ("model", model),
            ]
        )

    lr_clf = LogisticRegression(max_iter=1000)
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
    )

    pipelines = {
        "logistic_regression": build_pipeline(lr_clf),
        "xgboost": build_pipeline(xgb_clf),
        "mlp": build_pipeline(mlp_clf),
    }

    return pipelines

def train_all_models(data_path, target_col, test_size=0.2, random_state=42):
    df = load_raw_data(data_path)
    preprocessor = build_preprocessor(df, target_col)
    df[target_col] = df[target_col].replace({'no': 0, 'yes': 1})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns
    feature_names = list(numeric_features) + list(categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = build_models(preprocessor, feature_names)
    log = get_logger(__name__)
    results = {}

    for name, model in models.items():
        log.info(f"Entrenando modelo: {name}")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_auc_mean"] = scores.mean()
        results[name] = metrics

    return results
