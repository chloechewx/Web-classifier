from load_data import load_data
from preprocess import general_clean
from evaluate import evaluate_model
from report import save_comparison_report
from sklearn.model_selection import train_test_split,  GridSearchCV
import config

from models.logistic_regression import logistic_model
from models.random_forest_tree import random_forest_model
from models.xgboost_tree import xgboost_model
from models.catboost_tree import catboost_model


def main():
    df = load_data()
    df = general_clean(df)

    # Split X / y
    y = df[config.target_col]
    X = df.drop(columns=[config.target_col])

    # Identify numeric & categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = config.categorical_cols

    # Train/test split (stratify expects array)
    if config.stratify:
        stratify_arg = y
    else:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_arg,
    )

    results = {}
    metrics_by_model = {}

    # Logistic Regression
    if "logistic" in config.models_to_run:
        log_model = logistic_model(
            num_cols=num_cols,
            cat_cols=cat_cols,
            robust_cols=config.robust_scale_cols,
            standard_cols=config.standard_scale_cols
        )
            # Finetuning toggle in config
        if getattr(config, "tune_logistic", False):
            log_param_grid = getattr(config, "log_param_grid", {
                "model__C": [0.01, 0.1, 1.0],
                "model__class_weight": [None, "balanced"],
                })
    
            log_search = GridSearchCV(
                estimator=log_model,
                param_grid=log_param_grid,
                scoring="f1",
                cv=3,
                n_jobs=1,
                refit=True,
            )
            log_search.fit(X_train, y_train)
            log_model = log_search.best_estimator_
            print("\nLogistic Regression Using tuned model from GridSearchCV")
        else:

            print("\nLogistic Regression")
            log_model.fit(X_train, y_train)


        log_metrics = evaluate_model(
            "Logistic Regression", 
            log_model, 
            X_test, 
            y_test, 
            save_dir=config.result_dir,
            threshold=config.logistic_threshold,
            )
        results["logistic"] = log_model
        metrics_by_model["Logistic Regression"] = log_metrics

    # Random Forest
    if "random_forest" in config.models_to_run:
        rf_model = random_forest_model(
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        rf_model.fit(X_train, y_train)
        print("\nRandom Forest")
        random_forest_metrics = evaluate_model("Random Forest", rf_model, X_test, y_test, save_dir=config.result_dir)
        results["random_forest"] = rf_model
        metrics_by_model["Random Forest"] = random_forest_metrics


    # XGBoost
    if "xgboost" in config.models_to_run:
        xgb_model = xgboost_model(
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        
        # Finetuning toggle in config
        if getattr(config, "tune_xgboost", False):
            xgb_param_grid = getattr(config, "xgb_param_grid", {
                "model__n_estimators": [200, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [4,5],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            })

            xgb_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=xgb_param_grid,
                scoring="f1",
                cv=3,
                n_jobs=1,
                refit=True,
            )
            xgb_search.fit(X_train, y_train)
            xgb_model = xgb_search.best_estimator_
            print("\nXGBoost Using tuned model from GridSearchCV")
        else:
            print("\nXGBoost")
            xgb_model.fit(X_train, y_train)

        xgboost_metrics = evaluate_model(
            "XGBoost", 
            xgb_model, 
            X_test, 
            y_test, 
            save_dir=config.result_dir,
            threshold=config.xgboost_threshold,
            )
        results["xgboost"] = xgb_model
        metrics_by_model["XGBoost"] = xgboost_metrics



    #CatBoost
    if "catboost" in config.models_to_run:
        cb_model = catboost_model(cat_cols=cat_cols)
        cb_model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,  # CatBoost uses column names
        )
        print("\nCatBoost")
        catboost_metrics = evaluate_model("CatBoost", cb_model, X_test, y_test, save_dir=config.result_dir)
        results["catboost"] = cb_model
        metrics_by_model["CatBoost"] = catboost_metrics


    
    if metrics_by_model:
        print("\n=== Model Comparison (sorted by F1 score) ===")
        for model_name, m in sorted(
            metrics_by_model.items(),
            key=lambda kv: kv[1].get("f1", 0.0) if kv[1].get("f1") is not None else 0.0,
            reverse=True,
        ):
            print(
                f"{model_name:15s} | "
                f"F1={m['f1']:.4f}  "
                f"Roc-Auc{m['roc_auc']:.4f}  "
                f"Prec={m['precision']:.4f}  "
                f"Rec={m['recall']:.4f}"
                f"Acc={m['accuracy']:.4f}  "
            )

    # Optional markdown comparison report 
    if (
        getattr(config, "enable_report", False)
        and getattr(config, "enable_comparison_report", False)
    ):
        save_comparison_report(
            metrics_by_model=metrics_by_model,
            primary_metric="f1",
            result_dir=config.result_dir,
        )

    return results


if __name__ == "__main__":
    main()


