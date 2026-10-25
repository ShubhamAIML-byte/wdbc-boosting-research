import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

SEED = 42
EPSILON_OPT = 0.10
EPSILON_TEST = [0.0, 0.05, 0.10, 0.20, 0.30]

print("1. Loading WDBC Dataset...")
data = load_breast_cancer()
X = data.data
feature_names = list(data.feature_names)
y = (data.target == 0).astype(int) # 1 = Malignant, 0 = Benign

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

models = {
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, eval_metric="logloss",
        random_state=SEED, n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
        n_estimators=100, learning_rate=0.5, random_state=SEED
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.05,
        loss_function="Logloss", verbose=0,
        random_seed=SEED, thread_count=2,
        allow_writing_files=False
    )
}

names = list(models.keys())
K = len(names)

def make_attack(X_input, y_true, surrogate, epsilon):
    if epsilon == 0:
        return X_input.copy()
    p = np.clip(surrogate.predict_proba(X_input)[:, 1], 1e-7, 1 - 1e-7)
    grad = ((p - y_true)[:, None] * surrogate.coef_[0][None, :])
    return X_input + epsilon * np.sign(grad)

print("2. Computing 5-Fold OOF Predictions...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_clean = np.zeros((len(y_train), K))
oof_adv = np.zeros((len(y_train), K))

for fold, (itr, iva) in enumerate(cv.split(X_train_raw, y_train), 1):
    scaler_fold = StandardScaler()
    Xtr = scaler_fold.fit_transform(X_train_raw[itr])
    Xva = scaler_fold.transform(X_train_raw[iva])
    ytr, yva = y_train[itr], y_train[iva]

    surrogate_fold = LogisticRegression(max_iter=2000)
    surrogate_fold.fit(Xtr, ytr)
    Xva_adv = make_attack(Xva, yva, surrogate_fold, EPSILON_OPT)

    for j, name in enumerate(names):
        model = clone(models[name])
        model.fit(Xtr, ytr)
        oof_clean[iva, j] = model.predict_proba(Xva)[:, 1]
        oof_adv[iva, j] = model.predict_proba(Xva_adv)[:, 1]

print("3. Optimizing Ensemble Weights via SLSQP...")
def weighted_logloss(weights, probabilities, labels):
    p = np.clip(probabilities @ weights, 1e-7, 1 - 1e-7)
    return log_loss(labels, p, labels=[0, 1])

LAMBDA_CLEAN = 0.5
LAMBDA_ADV = 0.5

def objective(w):
    clean_loss = weighted_logloss(w, oof_clean, y_train)
    adv_loss = weighted_logloss(w, oof_adv, y_train)
    return LAMBDA_CLEAN * clean_loss + LAMBDA_ADV * adv_loss

opt = minimize(
    objective,
    x0=np.ones(K) / K,
    method="SLSQP",
    bounds=[(0, 1)] * K,
    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    options={"maxiter": 500, "ftol": 1e-10}
)
weights = opt.x.tolist()
equal_weights = [1.0 / K] * K
print(f"Optimized weights: {dict(zip(names, [round(w, 4) for w in weights]))}")

print("4. Fitting Models on Full Training Partition...")
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_raw)
Xte = scaler.transform(X_test_raw)

fitted_models = {}
for name in names:
    model = clone(models[name])
    model.fit(Xtr, y_train)
    fitted_models[name] = model

test_clean = np.column_stack([
    fitted_models[n].predict_proba(Xte)[:, 1] for n in names
])
p_equal = test_clean @ np.array(equal_weights)
p_optimized = test_clean @ np.array(weights)

def calc_metrics(y_true, p, label):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return {
        "model": label,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "confusion_matrix": cm
    }

clean_metrics = [calc_metrics(y_test, test_clean[:, j], name) for j, name in enumerate(names)]
clean_metrics.append(calc_metrics(y_test, p_equal, "Equal-Weight Voting"))
clean_metrics.append(calc_metrics(y_test, p_optimized, "Optimized Ensemble"))

print("5. Evaluating FGSM Transfer-based Adversarial Robustness...")
surrogate = LogisticRegression(max_iter=2000)
surrogate.fit(Xtr, y_train)

robustness_results = []
for eps in EPSILON_TEST:
    X_attack = make_attack(Xte, y_test, surrogate, eps)
    ind_p = np.column_stack([fitted_models[n].predict_proba(X_attack)[:, 1] for n in names])
    p_attack = ind_p @ np.array(weights)
    row = calc_metrics(y_test, p_attack, f"Epsilon {eps:.2f}")
    row["epsilon"] = eps
    robustness_results.append(row)

clean_acc = robustness_results[0]["accuracy"]
clean_auc = robustness_results[0]["roc_auc"]
for r in robustness_results:
    r["accuracy_drop_pp"] = round((clean_acc - r["accuracy"]) * 100, 2)
    r["auc_drop_pp"] = round((clean_auc - r["roc_auc"]) * 100, 2)

print("6. Calculating SHAP Explanations & Stability...")
def positive_class_shap(model, X_values):
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_values)
    except Exception:
        explainer = shap.Explainer(model.predict, X_values)
        values = explainer(X_values).values
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, :, 0]
    return values

shap_clean = {}
shap_perturbed = {}
X_shap_adv = Xte + np.random.default_rng(SEED).normal(0, 0.10, size=Xte.shape)

shap_stability = []
all_clean_importances = []
feature_instabilities = []

for name, model in fitted_models.items():
    sv_clean = positive_class_shap(model, Xte)
    sv_pert = positive_class_shap(model, X_shap_adv)
    shap_clean[name] = sv_clean
    shap_perturbed[name] = sv_pert

    imp_clean = np.abs(sv_clean).mean(axis=0)
    imp_pert = np.abs(sv_pert).mean(axis=0)
    all_clean_importances.append(imp_clean)

    diff = np.abs(sv_clean - sv_pert).mean(axis=0)
    feature_instabilities.append(diff)

    rho = float(spearmanr(imp_clean, imp_pert).statistic)
    k = 10
    top_clean = set(np.argsort(imp_clean)[-k:])
    top_pert = set(np.argsort(imp_pert)[-k:])
    overlap = float(len(top_clean & top_pert) / k)

    shap_stability.append({
        "model": name,
        "rank_correlation": rho,
        "top10_overlap": overlap
    })

aggregate_importance = np.average(
    np.vstack(all_clean_importances),
    axis=0,
    weights=weights
).tolist()

mean_instability = np.average(
    np.vstack(feature_instabilities),
    axis=0,
    weights=weights
).tolist()

shap_features = []
for i, f_name in enumerate(feature_names):
    shap_features.append({
        "feature": f_name,
        "importance": float(aggregate_importance[i]),
        "instability": float(mean_instability[i]),
        "xgboost_importance": float(all_clean_importances[0][i]),
        "adaboost_importance": float(all_clean_importances[1][i]),
        "catboost_importance": float(all_clean_importances[2][i])
    })
shap_features.sort(key=lambda x: x["importance"], reverse=True)

print("7. SHAP-Guided Retraining on Robust Subspace...")
unstable_sorted = sorted(range(len(mean_instability)), key=lambda idx: mean_instability[idx], reverse=True)
unstable_indices = unstable_sorted[:5] # Top 5 unstable features
masked_feature_names = [feature_names[idx] for idx in unstable_indices]
print(f"Top 5 masked unstable features: {masked_feature_names}")

Xtr_robust = Xtr.copy()
Xte_robust = Xte.copy()
Xtr_robust[:, unstable_indices] = 0
Xte_robust[:, unstable_indices] = 0

robust_fitted_models = {}
for name in names:
    model = clone(models[name])
    model.fit(Xtr_robust, y_train)
    robust_fitted_models[name] = model

retrained_robustness = []
for eps in EPSILON_TEST:
    X_attack_robust = make_attack(Xte_robust, y_test, surrogate, eps)
    ind_p = np.column_stack([robust_fitted_models[n].predict_proba(X_attack_robust)[:, 1] for n in names])
    p_attack_robust = ind_p @ np.array(weights)
    row = calc_metrics(y_test, p_attack_robust, f"Epsilon {eps:.2f}")
    row["epsilon"] = eps
    retrained_robustness.append(row)

print("8. Curating Representative Patient Test Cases...")
# Pick interesting samples from X_test_raw:
# Clear Benign, Clear Malignant, Borderline (prob ~ 0.45 - 0.55), Adversarially Flipped case
test_preds = p_optimized
test_df = pd.DataFrame(X_test_raw, columns=feature_names)
test_df["y_true"] = y_test
test_df["p_optimized"] = test_preds

# Benign with high confidence
benign_idx = int(np.where((y_test == 0) & (test_preds < 0.05))[0][0])
# Malignant with high confidence
malig_idx = int(np.where((y_test == 1) & (test_preds > 0.95))[0][0])
# Borderline cases
borderline_indices = np.where(np.abs(test_preds - 0.5) < 0.25)[0]
borderline_idx = int(borderline_indices[0]) if len(borderline_indices) > 0 else 0

patient_cases = [
    {
        "id": "WDBC-PT-104",
        "title": "Case 1: Benign Fibroadenoma",
        "description": "Typical benign FNA cytology with uniform small nuclei, smooth boundaries, low texture and small nuclear area.",
        "true_diagnosis": "Benign",
        "true_label": 0,
        "features": {f: float(X_test_raw[benign_idx, i]) for i, f in enumerate(feature_names)}
    },
    {
        "id": "WDBC-PT-249",
        "title": "Case 2: Invasive Ductal Carcinoma (Malignant)",
        "description": "High-grade malignant lesion presenting marked nuclear pleomorphism, high concavity, irregular boundaries and enlarged nuclear area.",
        "true_diagnosis": "Malignant",
        "true_label": 1,
        "features": {f: float(X_test_raw[malig_idx, i]) for i, f in enumerate(feature_names)}
    },
    {
        "id": "WDBC-PT-318",
        "title": "Case 3: Atypical Ductal Hyperplasia (Borderline)",
        "description": "Ambiguous nuclear features borderline between severe hyperplasia and carcinoma in situ; demonstrates ensemble utility in uncertain cases.",
        "true_diagnosis": "Malignant" if y_test[borderline_idx] == 1 else "Benign",
        "true_label": int(y_test[borderline_idx]),
        "features": {f: float(X_test_raw[borderline_idx, i]) for i, f in enumerate(feature_names)}
    }
]

# Compute feature statistics (min, max, mean, std) across entire WDBC dataset
feature_meta = {}
for i, f in enumerate(feature_names):
    col = X[:, i]
    benign_col = X[y == 0, i]
    malig_col = X[y == 1, i]
    feature_meta[f] = {
        "index": i,
        "name": f,
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(scaler.mean_[i]),
        "scale": float(scaler.scale_[i]),
        "benign_mean": float(benign_col.mean()),
        "malignant_mean": float(malig_col.mean())
    }

# Also serialize logistic surrogate for FGSM attack calculation in JS
surrogate_coef = surrogate.coef_[0].tolist()
surrogate_intercept = float(surrogate.intercept_[0])

# Export clean JSON payload
payload = {
    "dataset": {
        "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "source": "UCI ML Repository / Scikit-Learn",
        "total_samples": int(X.shape[0]),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "num_features": int(X.shape[1]),
        "malignant_count": int(y.sum()),
        "benign_count": int((y == 0).sum())
    },
    "feature_names": feature_names,
    "feature_metadata": feature_meta,
    "ensemble_weights": {
        "XGBoost": float(weights[0]),
        "AdaBoost": float(weights[1]),
        "CatBoost": float(weights[2])
    },
    "surrogate": {
        "coefficients": surrogate_coef,
        "intercept": surrogate_intercept
    },
    "clean_metrics": clean_metrics,
    "robustness_test": robustness_results,
    "shap_stability": shap_stability,
    "shap_features": shap_features,
    "shap_retrained_robustness": retrained_robustness,
    "masked_unstable_features": masked_feature_names,
    "patient_cases": patient_cases,
    "scaler": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
}

with open("model_data.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("\nSUCCESS: Exported model_data.json successfully!")
