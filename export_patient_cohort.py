import numpy as np
import pandas as pd
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42
data = load_breast_cancer()
X = data.data
feature_names = list(data.feature_names)
y = (data.target == 0).astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_raw)
Xte = scaler.transform(X_test_raw)

# Fit models
xgb = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    reg_lambda=1.0, eval_metric="logloss",
    random_state=SEED, n_jobs=-1
)
xgb.fit(Xtr, y_train)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
    n_estimators=100, learning_rate=0.5, random_state=SEED
)
ada.fit(Xtr, y_train)

cat = CatBoostClassifier(
    iterations=200, depth=5, learning_rate=0.05,
    loss_function="Logloss", verbose=0,
    random_seed=SEED, thread_count=2,
    allow_writing_files=False
)
cat.fit(Xtr, y_train)

surrogate = LogisticRegression(max_iter=2000)
surrogate.fit(Xtr, y_train)

# Weights from OOF SLSQP
w_xgb = 0.1935
w_ada = 0.0245
w_cat = 0.7820

# Let's export all 114 test patients with their features and model predictions
test_patients = []
p_xgb_all = xgb.predict_proba(Xte)[:, 1]
p_ada_all = ada.predict_proba(Xte)[:, 1]
p_cat_all = cat.predict_proba(Xte)[:, 1]
p_ens_all = p_xgb_all * w_xgb + p_ada_all * w_ada + p_cat_all * w_cat

for i in range(len(y_test)):
    t_label = "Malignant" if y_test[i] == 1 else "Benign"
    pred_label = "Malignant" if p_ens_all[i] >= 0.5 else "Benign"
    is_correct = (y_test[i] == (1 if p_ens_all[i] >= 0.5 else 0))
    test_patients.append({
        "index": i,
        "id": f"WDBC-PT-{1000 + i}",
        "true_label": int(y_test[i]),
        "true_diagnosis": t_label,
        "predicted_diagnosis": pred_label,
        "is_correct": bool(is_correct),
        "prob_xgboost": round(float(p_xgb_all[i]), 4),
        "prob_adaboost": round(float(p_ada_all[i]), 4),
        "prob_catboost": round(float(p_cat_all[i]), 4),
        "prob_ensemble": round(float(p_ens_all[i]), 4),
        "features": {feature_names[j]: round(float(X_test_raw[i, j]), 4) for j in range(len(feature_names))}
    })

print(f"Computed {len(test_patients)} test patients.")

# Export AdaBoost trees
ada_trees = []
for est, weight in zip(ada.estimators_, ada.estimator_weights_):
    tree = est.tree_
    ada_trees.append({
        "weight": float(weight),
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, :].tolist()
    })

# Export Decision Trees for XGBoost using dump
xgb_dump = xgb.get_booster().get_dump(dump_format='json')
print(f"XGB dump trees: {len(xgb_dump)}")

# Save cohort and trees to cohort_and_models.json
output_data = {
    "test_cohort": test_patients,
    "ada_trees": ada_trees,
    "xgb_trees_json": [json.loads(t) for t in xgb_dump],
}

with open("cohort_and_models.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f)

print("Exported cohort_and_models.json successfully!")
