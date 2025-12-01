
from catboost import CatBoostClassifier

def catboost_model(cat_cols, random_state=42):

    #No encode, no scaling
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        n_estimators=300,
        loss_function="Logloss",
        verbose=0,
        random_state=random_state,
    )
    return model