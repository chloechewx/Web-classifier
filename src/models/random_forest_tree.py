from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def random_forest_model(num_cols, cat_cols, random_state=42):

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
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )

    return model