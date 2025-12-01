from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def xgboost_model(num_cols, cat_cols, random_state=42):

    transformers = [
        ("num", "passthrough", num_cols),
    ]

    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
            )),
        ]
    )

    return model