import numpy as np
import pandas as pd
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

SEED = 42
data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_raw)
Xte = scaler.transform(X_test_raw)

cat = CatBoostClassifier(
    iterations=200, depth=5, learning_rate=0.05,
    loss_function="Logloss", verbose=0,
    random_seed=SEED, thread_count=2,
    allow_writing_files=False
)
cat.fit(Xtr, y_train)

# Save CatBoost to json
cat.save_model("cat_model.json", format="json")

with open("cat_model.json", "r") as f:
    cat_json = json.load(f)

# Extract only the minimal fields needed for scoring to keep data size lean
cat_compact = {
    "scale_and_bias": cat_json.get("scale_and_bias", [[1.0, 0.0]]),
    "trees": []
}

for t in cat_json.get("oblivious_trees", []):
    cat_compact["trees"].append({
        "splits": [{"feature": s["float_feature_index"], "border": float(s["border"])} for s in t.get("splits", [])],
        "leaf_values": [float(v) for v in t.get("leaf_values", [])]
    })

print(f"Compact CatBoost trees: {len(cat_compact['trees'])}")

# Update data.js
with open("model_data.json", "r") as f:
    model_data = json.load(f)

with open("cohort_and_models.json", "r") as f:
    cohort_data = json.load(f)

model_data["test_cohort"] = cohort_data["test_cohort"]
model_data["ada_trees"] = cohort_data["ada_trees"]
model_data["xgb_trees_json"] = cohort_data["xgb_trees_json"]
model_data["cat_model"] = cat_compact

categories = {
    "mean radius": ("Nuclear Size", "Mean distance from center to perimeter points", "mm"),
    "mean texture": ("Cell Texture", "Standard deviation of gray-scale values across nucleus", "intensity units"),
    "mean perimeter": ("Nuclear Size", "Mean perimeter of cell nuclei", "mm"),
    "mean area": ("Nuclear Size", "Mean area of cell nuclei", "mm²"),
    "mean smoothness": ("Cell Boundary", "Mean local variation in radius lengths", "index"),
    "mean compactness": ("Nuclear Shape", "Perimeter² / Area - 1.0 (shape irregularity)", "ratio"),
    "mean concavity": ("Nuclear Shape", "Severity of concave portions of the contour", "contour depth"),
    "mean concave points": ("Nuclear Shape", "Number of concave portions of the contour", "count ratio"),
    "mean symmetry": ("Nuclear Shape", "Nuclear symmetry across central axis", "symmetry index"),
    "mean fractal dimension": ("Cell Boundary", "'Coastline approximation' - 1 (perimeter complexity)", "fractal index"),
    
    "radius error": ("Measurement Variance", "Standard error for the mean distance from center", "SE mm"),
    "texture error": ("Measurement Variance", "Standard error for gray-scale intensity variation", "SE"),
    "perimeter error": ("Measurement Variance", "Standard error for nuclear perimeter", "SE mm"),
    "area error": ("Measurement Variance", "Standard error for nuclear area (high instability feature)", "SE mm²"),
    "smoothness error": ("Measurement Variance", "Standard error for local radius variations", "SE"),
    "compactness error": ("Measurement Variance", "Standard error for nuclear compactness", "SE"),
    "concavity error": ("Measurement Variance", "Standard error for contour concavity severity", "SE"),
    "concave points error": ("Measurement Variance", "Standard error for concave contour points count", "SE"),
    "symmetry error": ("Measurement Variance", "Standard error for nuclear symmetry", "SE"),
    "fractal dimension error": ("Measurement Variance", "Standard error for fractal dimension", "SE"),

    "worst radius": ("Worst / Extreme Cellular Features", "Largest ('worst') mean radius across cell sample", "mm"),
    "worst texture": ("Worst / Extreme Cellular Features", "Largest ('worst') gray-scale standard deviation", "intensity units"),
    "worst perimeter": ("Worst / Extreme Cellular Features", "Largest ('worst') nuclear perimeter", "mm"),
    "worst area": ("Worst / Extreme Cellular Features", "Largest ('worst') nuclear area", "mm²"),
    "worst smoothness": ("Worst / Extreme Cellular Features", "Largest ('worst') local radius variation", "index"),
    "worst compactness": ("Worst / Extreme Cellular Features", "Largest ('worst') nuclear compactness ratio", "ratio"),
    "worst concavity": ("Worst / Extreme Cellular Features", "Largest ('worst') contour concavity severity", "depth"),
    "worst concave points": ("Worst / Extreme Cellular Features", "Largest ('worst') count of concave contour points", "count ratio"),
    "worst symmetry": ("Worst / Extreme Cellular Features", "Most asymmetric nuclear measurement observed", "index"),
    "worst fractal dimension": ("Worst / Extreme Cellular Features", "Highest ('worst') perimeter complexity index", "index"),
}

for fname, meta in model_data["feature_metadata"].items():
    if fname in categories:
        meta["category"] = categories[fname][0]
        meta["description"] = categories[fname][1]
        meta["unit"] = categories[fname][2]

js_content = "window.WDBC_DATA = " + json.dumps(model_data) + ";\n"
with open("data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"Updated data.js with all 3 full models! File size: {len(js_content)} bytes")
