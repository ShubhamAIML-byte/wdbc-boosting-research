# 🔬 WDBC ResilientBoost: Reliability-Aware Boosting Oncology AI

> **Interactive Clinical Platform & Deployment for the Wisconsin Diagnostic Breast Cancer (WDBC) Boosting Research Pipeline.**  
> Operationalizing the Google Colab Research Pipeline: [WDBC Boosting Research Colab](https://colab.research.google.com/drive/1B4RYPAVQzh70srBgxTj9A6OunOlJCOjh?usp=sharing)

---

## 🌟 Executive Summary

This web application operationalizes a state-of-the-art **Reliability-Aware Tri-Boosting Machine Learning Ensemble** for breast cancer fine needle aspirate (FNA) cytopathology triage. 

Unlike conventional ensembles that optimize solely on clean validation error, this architecture couples **CatBoost**, **XGBoost**, and **AdaBoost** through an **Out-of-Fold (OOF) SLSQP objective** balancing both clean and adversarial cross-entropy losses, backed by **SHAP explanation stability** analysis and **adversarially hardened subspace retraining**.

The entire machine learning inference engine—including 200 CatBoost oblivious decision trees, 200 XGBoost depth-3 trees, 100 AdaBoost estimators, FGSM surrogate gradient perturbations, and local SHAP feature attributions—runs **100% natively in modern client-side JavaScript** with sub-millisecond execution latency and zero external server dependencies.

---

## 🚀 1-Click Vercel Deployment

This project is configured out-of-the-box for instant deployment to [Vercel](https://vercel.com):

### Option A: Deploy via Vercel CLI
```bash
# 1. In the project directory:
cd "d:\boosting wdbc"

# 2. Deploy to production:
vercel --prod
```

### Option B: Deploy via GitHub & Vercel Dashboard
1. Initialize git and push to your GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of WDBC ResilientBoost"
   git branch -M main
   git remote add origin https://github.com/<your-username>/boosting-wdbc.git
   git push -u origin main
   ```
2. Navigate to [vercel.com/new](https://vercel.com/new).
3. Import the repository and click **Deploy**.
4. Vercel automatically deploys the static application with global edge CDN distribution, zero cold starts, and automated SSL.

---

## 💻 Local Development

Run the zero-dependency local development server:

```bash
# Start local development server on http://localhost:3000
npm run dev
# OR
node server.js
```

---

## 📊 Research Highlights & Empirical Findings

### 1. Model Performance (Untouched 114 Test Cases)
| Model Architecture | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost** (100 Trees) | 97.37% | 100.0% | 92.86% | 100.0% | 96.30% | 0.9878 |
| **XGBoost** (200 Trees) | 96.49% | 100.0% | 90.48% | 100.0% | 95.00% | 0.9940 |
| **CatBoost** (200 Trees) | 96.49% | 100.0% | 90.48% | 100.0% | 95.00% | **0.9993** |
| **Equal-Weight Voting** | 96.49% | 100.0% | 90.48% | 100.0% | 95.00% | 0.9937 |
| **SLSQP-Optimized Ensemble** | 96.49% | **100.0%** | 90.48% | **100.0%** | 95.00% | **0.9983** |

*Notice: The optimized ensemble achieved **0 False Positives** (100% Specificity) on the held-out test partition.*

---

### 2. OOF Reliability-Aware SLSQP Weight Optimization
The ensemble weights are solved via Sequential Least Squares Programming (SLSQP) across 5-fold cross-validation:
$$\min_{w} \; \lambda_{\text{clean}} \mathcal{L}_{\text{clean}}(y, P_{\text{clean}} w) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(y, P_{\text{adv}} w)$$
$$\text{subject to} \quad \sum_{i=1}^3 w_i = 1, \quad w_i \ge 0$$

* **CatBoost**: `78.20%` (Highest weight due to oblivious tree symmetry and superior resistance to transfer attack gradients)
* **XGBoost**: `19.35%`
* **AdaBoost**: `2.45%`

---

### 3. FGSM Transfer Adversarial Robustness
Simulated FNA microscopic staining artifacts and gradient transfer stress testing:
$$x' = x + \epsilon \cdot \operatorname{sign}\left(\nabla_x \mathcal{L}_{\text{surrogate}}\right)$$

| Perturbation Budget ($\epsilon$) | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Accuracy Drop |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\epsilon = 0.00$ (Clean) | 96.49% | 100.0% | 90.48% | 100.0% | 0.9500 | 0.9983 | 0.00 pp |
| $\epsilon = 0.05$ | 95.61% | 100.0% | 88.10% | 100.0% | 0.9367 | 0.9934 | -0.88 pp |
| $\epsilon = 0.10$ | 93.86% | 97.30% | 85.71% | 98.61% | 0.9114 | 0.9815 | -2.63 pp |
| $\epsilon = 0.20$ | 86.84% | 86.49% | 76.19% | 93.06% | 0.8101 | 0.9302 | -9.65 pp |
| $\epsilon = 0.30$ | 79.82% | 74.36% | 69.05% | 86.11% | 0.7160 | 0.8439 | -16.67 pp |

---

### 4. SHAP Stability & Explanation Robustness
* **Spearman Rank Correlation**: $\rho = 0.9982$
* **Top-10 Feature Overlap**: $96.7\%$
* **Top 5 Diagnostic Drivers**: `worst texture`, `worst concave points`, `mean concave points`, `worst area`, `area error`.

---

### 5. SHAP-Guided Retraining (Robust Feature Subspace)
By identifying the top 5 unstable features (`area error`, `worst area`, `worst texture`, `mean concave points`, `worst radius`) and retraining the ensemble on the robust subspace:
* Clean test accuracy improves to **97.37%** (Sensitivity **92.86%**)
* Severe attack accuracy at $\epsilon = 0.30$ improves from **79.82% to 84.21%** (**+4.39 pp robustness gain**)!

---

## 🛠️ Project Structure

```
d:/boosting wdbc/
├── index.html                   # Master single-page application
├── css/
│   └── styles.css               # Vanilla CSS clinical glassmorphic design system
├── js/
│   ├── engine.js                # Client-side ML inference engine (AdaBoost, XGBoost, CatBoost, FGSM)
│   └── app.js                   # UI controller, state management, charts, and report generator
├── data.js                      # Pre-trained models, OOF weights, metadata, cohort & benchmarks
├── server.js                    # Zero-dependency local development server
├── vercel.json                  # Vercel production deployment configuration
├── package.json                 # Project manifest & npm scripts
├── train_and_export.py          # Python pipeline to train, evaluate, and export models
├── export_catboost.py           # CatBoost tree exporter and data assembler
├── wdbc_boosting_research.ipynb # Downloaded Google Colab research notebook
└── README.md                    # Comprehensive documentation
```

---

## 📄 License
MIT License. Built for clinical and machine learning research demonstration.
