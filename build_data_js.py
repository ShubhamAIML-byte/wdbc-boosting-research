import json

with open("model_data.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)

with open("cohort_and_models.json", "r", encoding="utf-8") as f:
    cohort_data = json.load(f)

# Add feature categories and human friendly descriptions
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

# Attach cohort and trees
model_data["test_cohort"] = cohort_data["test_cohort"]
model_data["ada_trees"] = cohort_data["ada_trees"]
model_data["xgb_trees_json"] = cohort_data["xgb_trees_json"]

# Write as data.js
js_content = "window.WDBC_DATA = " + json.dumps(model_data) + ";\n"
with open("data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"Generated data.js ({len(js_content)} bytes).")
