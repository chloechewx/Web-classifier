
#---database loading config---
database = "data/phishing.db"
table_name = "phishing_data"

#---target column---
target_col = "label"

#---training settings---
test_size = 0.2
random_state = 42
stratify = True 

#---Specific categorial columns--
categorical_cols = ["Industry", "HostingProvider"]

#--- Scaling for Logistic model ---
robust_scale_cols = [
    "LineOfCode", "LargestLineLength", "NoOfPopup",
    "NoOfiFrame", "NoOfImage", "NoOfSelfRef", "NoOfExternalRef"
]
standard_scale_cols = [
    "DomainAgeMonths"
]

#---Choosing models to run---
models_to_run = [
    "logistic"
    ,"random_forest"
    ,"xgboost"
    ,"catboost"
    ]


# ---Model tuning toggle--
# tune_logistic = True 
# tune_xgboost = True 

# ---Decision thresholds--- default is 0.50
# logistic_threshold = 0.45
# xgboost_threshold = 0.40
logistic_threshold = 0.50
xgboost_threshold = 0.50

#---Reporting option, will affect evaluate.py file---
enable_report = True
enable_comparison_report = True
result_dir = "results"




