from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from .feature_selection import HybridFeatureSelector


def build_preprocessor(df, target_col: str):
    """
    Devuelve:
    - preprocessor: ColumnTransformer
    - feature_name_extractor: función que devuelve los nombres limpios
    """
    
    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OrdinalEncoder(handle_unknown="error"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
