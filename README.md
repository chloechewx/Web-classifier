# Phishing vs Legitimate Website Classification 

Full Name: Chew Jia Xuan

Email: Chloe.jiaxuan@gmail.com

Author: Github.com/chloechewx

---------------------------------------------
# Overview of folder structure
```
root directory
├── .github # Github configuration files
├── src # ML pipeline source scripts
│ ├── config.py # Central pipeline config and toggles
│ ├── load_data.py # Load dataset from SQLite database
│ ├── preprocess.py # Global dataset cleaning 
│ ├── train.py # Train/test split and model execution
│ ├── evaluate.py # Compute all model metrics
│ ├── report.py # Generate markdown report
│ └── models 
│ ├── randomforest.py
│ ├── logistic.py
│ ├── xgboost.py
│ └── catboost.py
├── eda.ipynb # Exploratory Data Analysis notebook
├── requirements.txt # Python dependencies list
└── README.md # Project documentation
```
## Executing the pipeline

```
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
bash run.sh

# Alternatively you can run:
python src/train.py
```
## Adjust pipeline parameters
Modify any setting in config.py including:
- Model toggles (which models to train)
- Finetuning toggles (enable/disable)
- Decision thresholds 

Example: **Config.py**
```
models_to_run = [
    "logistic"
    ,"random_forest"
    ,"xgboost"
    ,"catboost"
    ]

tune_logistic = True

enable_report = True
enable_comparison_report = True
```
## ML Pipeline Logical Flow

```
SQLite DB
   ↓
load_data.py  → Loads dataset into memory  
   ↓
preprocess.py → Global cleaning, missing flags etc 
   ↓
train.py     → Train/test split and model training execution  
   ↓
evaluate.py  → Generate F1, ROC-AUC, Confusion Matrix, Classification Report  
   ↓
report.py    → Convert evaluation into readable markdown report (optional toggle)
```
## EDA Summary (Key Findings & Pipeline Choices)

 - Extreme outliers exist in 7 columns but contextually plausible, as both phishing and legitimate sites can contain large values. Hence, Tree based model were chosen to conduct modeling as it is Less sensitive to unscaled extreme values

 - Null values mostly exist in `label = 0 cases`. Instead of dropping it, `LineOfCode_missing` is added to preserve signal.

 - Negative `NoOfImage values` exist. 
 The negative value can be found in both phishing and legitimate website labels. It was then clipped to 0 to keep the data valid for model training.


 - Categorical column inconsistencies ('Ecommerce ' vs 'Ecommerce'). Prevent category duplicated by cleaning whitespace before encoding. 

## Feature Processing

| Attributes | Preprocess Applied | Reason |
| ---------- | --------------- | ------ |
| LineOfCode         | Median imputation + Created Missing flag. *Only for LR: Robust scaling*| Prevent fake correlations, preserve missing-code signal |
| LargestLineLength  | *Only for LR: Robust scaling* | Used as-is (real extreme values acceptable)
| NoOfURLRedirect    | None | Website behaviour signal |
| NoOfSelfRedirect   | None | Website behaviour signal |
| NoOfPopup          | *Only for LR: Robust scaling* | Website behaviour signal |
| NoOfiFrame         | *Only for LR: Robust scaling* | Website behaviour signal |
| NoOfImage          | Clip negatives to 0. *Only for LR: Robust scaling* | Maintain valid numeric domain |
| NoOfSelfRef        | *Only for LR: Robust scaling* | Website behaviour signal |
| NoOfExternalRef    | *Only for LR: Robust scaling* | Website behaviour signal |
| Robots             | None | Website contains robot.txt signal |
| IsResponsive       | None | Website behaviour signal
| Industry           | Whitespace strip before one-hot encoding. One-hot encoded for selected models | Avoid duplicate categories |
| DomainAgeMonths    | *Only for LR: Standard scaling* | No extreme values, Older domains generally safer signal |
| HostingProvide     | One-hot encoded for selected models | Improve classifier learning from source type |
| label              | Target column: Binary | Classification ground truth |


# Model Choice

The dataset contains numerous outliers and extreme numeric values, which are not errors but genuine characteristics that may appear in both legitimate and phishing websites.
The models were selected with the goal of evaluating which algorithm can remain stable and effective despite extreme values, without performance distortion or training failure.

- **Logistic Regression (Baseline)**: Linear classifier chosen for its interpretability and computational efficiency. It provides a clear reference for model behavior and enables direct inspection of feature coefficients, particularly useful for explaining influential signals such as engineered missing flags.

- **Random Forest** → A bagging based ensemble selected to test model stability against non-linear and outlier-heavy inputs.
- It also does not require numeric scaling, further protecting it from outlier distortions.

- **XGBoost** → A gradient boosted tree ensemble included to evaluate progressive learning capability and inference speed, able to have a controlled finetuning.

- **CatBoost** → A boosting model selected primarily for its native categorical processing and output of probability-based evaluations. 

# Model Evaluation

### Metrics Priority:
**F1**: As the objective is not simply minimising errors, but ensuring prediction trustworthiness. F1 captures both precision and recall. Since false negatives are operationally risky, F1 ensures recall is not ignored, but also prevents the model from overwhelming sensitivity that would severely hurt precision.

**ROC-AUC**: Measures the model’s ability to rank and separate phishing vs legitimate websites across all possible thresholds, independent of extreme raw values. It is also useful for adjusting model sensitivity after training.

**Confusion Matrix**: For tuning diagnostic purpose. Ensures mistakes are measured by direction and risk, not just totals, so tuning decisions can reduce False negatives and maintain safety.

## Observations Across 4 Models

**XGBoost**:
- Achieved the highest F1 (F1 = 0.8533), nearly identical to CatBoost (F1 ≈ 0.8530).
- Confusion Matrix:
  * **TN = 795, FP = 149**
  * **FN = 185, TP = 971**
- Although F1 and ROC-AUC were strong, the false negatives (185 FN) required attention, as missed phishing cases are operationally risky in security systems.
- Feature importances were small and fragmented. `LineOfCode_missing` flag is high, while the others features are smaller. It is not using many phishing signals.

**CatBoost**
- Achieved the highest ROC-AUC (ROC-AUC = 0.8935), followed closely by logistic regression (ROC-AUC = 0.8910)
- Confusion Matrix:
  * **TN = 805, FP = 139** (fewest FP at the cost of higher False negative)
  * **FN = 193, TP = 963**
- Maintained strong precision and F1, but FN was higher than XGBoost.
- `LineOfCode_missing` flag feature importance is super high(55.6805) also `HostingProvider`(6.2654) and `LineOfCode`(5.9455).

**Random Forest**
- Delivered the best recall (fewest missed phishing sites, FN = 166) but produced the highest false positives (FP = 201), meaning many safe sites would be incorrectly flagged. In a consumer-facing phishing warning system, this can harm trust, as users may start ignoring warnings when too many alerts are raised.
- Confusion Matrix:
  * **TN = 745, FP = 201** 
  * **FN = 166, TP = 990**
- Its top features were more evenly distributed across missing flags and real structural features, showing good robustness.

**Logistic Regression**
- Has a better balanced recall (180 FN) and FPs (177 FP), competable with boosting model as it's F1 (F1 = 0.8454) and ROC-AUC is only slightly lower. Can be taken into consideration for deploying small and simple model.
- Confusion Matrix:
  * **TN = 767, FP = 177** 
  * **FN = 180, TP = 976**
- Its weights leaned heavily on LineOfCode_missing (~6.65), indicating model dependence on a single missingness signal, which could reduce robustness if future data contained fewer or no missing flags. `LineOfCode_missing` flag (6.652) is very high, followed by very specifc feature `HostingProvider_Azure`(1.4342) and `HostingProvider_Google Cloud`(1.4042).
 
## Models to Finetune
Because all 4 models showed strong but very similar score ranges, I moved from default runs to principled cross-validated tuning on 2 contrasting models to balance performance and deployment feasibility.

**Model pairing logic**: The goal was not only to improve model scores, but also to ensure both detection reliability and practical deployment feasibility for a consumer-facing phishing browser extension, where the model must make predictions quickly and safely before a website fully loads.

-   **Logistic Regression** : Selected as a contrasting benchmark, because it is lightweight, very fast to train, and produces clear, auditable feature coefficients, which help explain how website attributes contribute to phishing detection. 
-   **XGBoost**: Competitive baseline performance while offering fast inference latency and strong capacity for non-linear feature interaction learning, which is useful when phishing signals overlap with legitimate cases. Unlike bagging ensembles, XGBoost also allows internal sensitivity shaping via parameters, giving it more tuning control in a compact search space.

This pairing enables comparison between linear decision boundaries (logistic) and boosted decision rule learning (XGBoost).

CatBoost was not chosen for first-round tuning because, although its discrimination metrics were strong, it had slower inference and less transparent feature attribution, since categorical influence is learned internally and not exposed explicitly in coefficient space.

### Finetuned model evaluation
In the initial baseline, both Logistic Regression and XGBoost were trained using default hyperparameters, without cross-validated optimisation. This means the models performed well, but were not explicitly tuned for phishing-specific prediction reliability.

To improve performance and gain control over the **precision–recall trade-off**, the following enhancements were introduced in both models:

- **GridSearchCV with 3-fold cross-validation** to ensure robust parameter selection .  
- **F1 scoring objective** to optimise predictive reliability instead of accuracy alone.
- **Regularisation and model-complexity controls** to reduce over-confidence on any single feature.
- **Lower evaluation thresholds** to reduce false negatives, which are more critical for phishing detection than false positives.
    
Both models retained strong discrimination ability (ROC-AUC ≈ 0.89), confirming that threshold-based sensitivity adjustments were justified post-training.

### **Logistic Regression**
Changes made during fine-tuning:

- Reduced **C (inverse regularisation strength)** to smaller values [0.01, 0.1, 1.0] to apply **stronger L2 regularisation**, making it more general and less likely to overfit.
    
- Evaluated multiple **class-weighting behaviours** (None, "balanced", {1:1.2}) to support phishing detection without allowing dominance by extreme weighting.
    
- Grid search used **F1 scoring**, ensuring recall improvement was balanced against precision loss.
    
- Applied **decision threshold = 0.45** during evaluation to improve sensitivity safely **without retraining the model**.
    
#### Impact after tuning:

- **F1 improved slightly** due to better recall balance
    
- **False negatives reduced** from **180 to 170**, improving operational safety
    
- The regularisation changes achieved the goal of reducing over-reliance on `LineOfCode_missing` weighing 
4.44, lesser than before 6.65.
    
- Other features (hosting providers, domain age, reference counts) now contribute **more proportionally**. This leads to a more balanced and interpretable model, which is less likely to fail if the missingness pattern changes in future data.

### **XGBoost Model**
Changes made during fine-tuning:
- Tested more trees [200, 300] and learning rates [0.05, 0.1] to explore recall safe behaviours
    
- Introduced **slower learning behaviour** (learning_rate = 0.05, n_estimators ≤ 300) to support incremental split learning on overlapping phishing cases
    
- Limited **tree depth = 4–5** to reduce overfitting on noisy or overly specific legit/phish interaction patterns
      
- Reduced **evaluation threshold to 0.40** to improve recall post-training without re-fitting

#### Impact after tuning:

- More balanced confusion matrix compare to before, FN reduced from 185 to 167 but the trade off is False positives increased from 149 to 175. 

- **F1 dropped marginally** (0.8533 → 0.8526), indicating recall improvement offset the minor precision trade off

- **Feature importance became more distributed** across real split-gain signals instead of exploding on `LineOfCode_missing`

## Conclusion and Deployment Considerations
Both finetuned models (Logistic & XGBoost) demonstrated consistently high phishing-legitimacy discrimination power (ROC-AUC ≈ 0.89) and reliable classification behavior Logistic Regression provided a fast, transparent, and interpretable baseline, while XGBoost is able to shape and deploy phishing-risk warnings due to its ability to learn non-linear feature interactions, produce higher true positives, reduce false negatives, and operate with low prediction latency, which is a key requirement for browser extension use.

Model size and inference latency were primary factors in model selection. XGBoost was chosen for fine-tuning and deployment instead of CatBoost because of its significantly faster processing and prediction speed, which is important in browser-based phishing defence where warnings must appear before a webpage fully loads. The dataset used was of medium size (10500 x 15), making XGBoost suitable without needing to downgrade to lighter algorithms at this point. Alternate boostign model, LightGBM can be considered later if the dataset becomes much larger and faster training or inference throughput is required.

Random Forest was not chosen despite having the lowest FN and best recall, because it produced a high number of false positives (201 FP). In a consumer-facing browser extension, excessive incorrect warnings can lead to alert fatigue. Users may start ignoring phishing prompts when too many safe websites are wrongly flagged, reducing overall trust in the product. Since this model will be deployed to a general audience, the priority was not maximising recall alone, but ensuring warnings feel reliable, timely, and not spam-like, while still maintaining a strong F1 performance.

Rapid evolution of phishing behaviour was another core consideration in model selection. As legitimate websites continue to adopt advanced web technologies, phishing websites also evolve to mimic real world behaviour more convincingly. This creates a need for models that can be periodically retrained while remaining flexible in real-time sensitivity tuning. Although CatBoost and Random Forest support probability outputs, XGBoost and Logistic Regression were preferred for finetuning due to their faster inference latency and transparent feature learning behaviour. Both models also enable post training threshold adjustments, allowing warning sensitivity to be recalibrated across retraining cycles without rebuilding learned model weights, which is especially valuable in deployment contexts such as browser extensions where latency, user trust, and false-negative auditing are critical.

----------------------------------------------------
