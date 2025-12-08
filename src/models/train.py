from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .data.load_data import load_raw_data
from .features.preprocessing import build_preprocessor, build_feature_selector
from .evaluation.metrics import evaluate_model
from .utils.logging import get_logger

def build_models(preprocessor):
    selector = build_feature_selector()

    lr_clf = LogisticRegression(max_iter=1000)
    xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                           subsample=0.9, colsample_bytree=0.8, eval_metric="logloss")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                           batch_size=256, learning_rate_init=1e-3, max_iter=50,
                           early_stopping=True)

    pipelines = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", lr_clf)
        ]),
        "xgboost": Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", xgb_clf)
        ]),
        "mlp": Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", mlp_clf)
        ])
    }

    return pipelines

def train_all_models(data_path, target_col, test_size=0.2, random_state=42):
    df = load_raw_data(data_path)
    preprocessor = build_preprocessor(df, target_col)
    df[target_col] = df[target_col].replace({'no': 0, 'yes': 1})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = build_models(preprocessor)
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
