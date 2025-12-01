from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
    )
from report import save_markdown_report, extract_feature_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import config


def evaluate_model(
        name, 
        model, 
        X_test, 
        y_test, 
        save_dir=None, 
        threshold: float = 0.5,
        ):

    y_proba = None
    roc_auc = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        # To apply custom threshold
        y_pred = (y_proba >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    else:
        print("ROC AUC Score: Not supported by this model")

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    accuracy  = accuracy_score(y_test, y_pred)


    print(f"F1 Score  : {f1:.4f}")
    print(f"Roc-Auc  : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    
    print("-" * 40)

    print("\nClassification Report:")
    cls_report = classification_report(y_test, y_pred)
    print(cls_report)
    print("-" * 40)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    cm_image_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.lower().replace(" ", "_")
        cm_image_path = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")
        plt.savefig(cm_image_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to: {cm_image_path}")

    plt.close()

    metrics = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

    # Feature importance / weights
    feature_importances = None
    if getattr(config, "enable_feature_importance", True):
        try:
            feature_scores = extract_feature_importance(model, X_test)
            if feature_scores:
                feature_importances = [(col, score) for col, score in feature_scores]
        except Exception as e:
            print(f"Could not extract feature importance for {name}: {e}")
            feature_importances = None


    # Optional report MD file creation
    if getattr(config, "enable_report", False):
        result_dir = config.result_dir
        save_markdown_report(
            name=name,
            metrics=metrics,
            cls_report=cls_report,
            confusion_matrix=cm,
            cm_image_path=cm_image_path,
            result_dir=result_dir,
            feature_importances=feature_importances
        )


    return metrics

    

