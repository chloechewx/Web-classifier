import os
from datetime import datetime
from tabulate import tabulate
import numpy as np

#To build markdown report file
def save_markdown_report(
    name,
    metrics,
    cls_report,
    confusion_matrix,
    cm_image_path=None,
    result_dir="results",
    feature_importances=None
):
    os.makedirs(result_dir, exist_ok=True)

    safe_name = name.lower().replace(" ", "_")
    result_path = os.path.join(result_dir, f"{safe_name}_report.md")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# Evaluation Report – {name}\n")
    lines.append(f"_Generated on: {timestamp}_\n")

    # Summary metrics
    lines.append("## Summary Metrics\n")
    roc_auc = metrics.get("roc_auc", None)

    lines.append(f"- **F1 Score**: {metrics['f1']:.4f}")

    if roc_auc is not None:
        lines.append(f"- **ROC AUC**: {roc_auc:.4f}")
    else:
        lines.append("- **ROC AUC**: Not supported by this model")
    
    lines.append(f"- **Precision**: {metrics['precision']:.4f}")
    lines.append(f"- **Recall**: {metrics['recall']:.4f}")
    lines.append(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")

    # Classification report
    lines.append("## Classification Report\n")
    lines.append("```text")
    lines.append(cls_report.strip())
    lines.append("```\n")

    #  Confusion matrix values
    lines.append("## Confusion Matrix (Values)\n")
    lines.append("```text")
    lines.append(str(confusion_matrix))
    lines.append("```\n")

    # Confusion matrix image
    if cm_image_path is not None:
        lines.append("## Confusion Matrix (Image)\n")
        lines.append(f"![Confusion Matrix]({cm_image_path})\n")

    # Feature importance / weights section
    if feature_importances:
        lines.append("## Feature Importance / Weights\n")
        lines.append(
            "The section below summarizes the top 10 importance scores learned by the model. "
            "These scores represent how strongly each feature influenced the phishing prediction.\n"
        )

        feat_table = [[feat, round(score, 4)] for feat, score in feature_importances[:10]]
        feat_table_md = tabulate(feat_table, headers=["Feature", "Importance"], tablefmt="github")

        lines.append(feat_table_md)
        lines.append("")  # Blank line for spacing



    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved evaluation report to: {result_path}")
    return result_path



def save_comparison_report(
        metrics_by_model,
        primary_metric="f1",
        result_dir="results",
        filename="model_comparison.md"
):
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, (filename))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    if len(metrics_by_model) <= 1:
        print("Not enough models for comparison report (need at least 2).")
        return None
    
    table_data = []
    for model_name, m in metrics_by_model.items():
        table_data.append([
            model_name,
            round(m.get("f1") or 0, 4),
            round(m.get("roc_auc") or 0, 4),
            round(m.get("precision") or 0, 4),
            round(m.get("recall") or 0, 4),
            round(m.get("accuracy") or 0, 4)
        ])

    headers = ["Model", "F1", "ROC-AUC", "Precision", "Recall", "Accuracy"]
    comparison_table = tabulate(table_data, headers=headers, tablefmt="github")

    lines = []
    lines.append("# Model Comparison\n")
    lines.append(f"_Generated on: {timestamp}_\n")
    lines.append("```\n" + comparison_table + "\n```\n")


    # Sort models by chosen primary metric
    sorted_items = sorted(
    metrics_by_model.items(),
    key=lambda kv: kv[1].get(primary_metric, 0.0) or 0.0,
    reverse=True,
)
    
    lines.append("\n## Ranking (by " + primary_metric.upper() + ")\n")
    for rank, (model_name, m) in enumerate(sorted_items, start=1):
        lines.append(
            f"{rank}. **{model_name}** – {primary_metric.upper()}={m[primary_metric]:.4f}, "
            f"ROC-AUC={m['roc_auc']:.4f}"
        )


    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Saved model comparison report to: {result_path}")
    return result_path



def extract_feature_importance(model, X_sample):

    base_model = model
    preprocessor = None

    # Pipeline models keep preprocessing under this step name
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocess", None)  
        base_model = model.named_steps.get("model", base_model)

    values = None

    # Tree models (XGBoost, Random Forest)
    if hasattr(base_model, "feature_importances_"):
        values = np.asarray(base_model.feature_importances_)

    # Linear models (Logistic Regression)
    elif hasattr(base_model, "coef_"):
        coef = np.asarray(base_model.coef_)
        values = np.abs(coef[0])

    # If no feature values found, return None
    if values is None:
        return None

    values = values.ravel()
    n_features = len(values)

    # Get the actual transformed column names from the encoder/scaler
    feature_names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw_names = preprocessor.get_feature_names_out()
            # remove internal prefixes
            feature_names = [n.split("__")[-1] for n in raw_names]
        except:
            feature_names = None

    # Fallback to your actual dataset columns only if pipeline did NOT transform anything
    if feature_names is None:
        cols = list(X_sample.columns)
        if len(cols) == n_features:
            feature_names = cols

    # final fallback — only runs if everything else fails
    if feature_names is None:
        feature_names = [f"{col}" for col in X_sample.columns]

    # Pair names with importance scores and sort
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs
