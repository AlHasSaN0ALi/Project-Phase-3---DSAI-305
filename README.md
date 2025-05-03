# Project-Phase-3---DSAI-305
## Ahassan Readme file 
# 📘 Project: Interpretability & Explainability in AI
**Course:** DSAI 305  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Goal:** Build interpretable machine learning models for binary classification (Malignant vs Benign) and apply explainability techniques (XAI) to understand model decisions.

---

## 🧪 1. EDA, Feature Engineering, and Selection
**Notebook:** `EDA_FeatureEngineering_Alhassan_DASI_305_Phase3.ipynb`

### 🔍 Key Steps:
- Checked data balance and feature distributions.
- Correlation heatmap to identify redundant features.
- Used **three feature selection techniques**:
  - Chi-Square Test
  - ANOVA F-Test
  - Information Gain (Mutual Information)
- Selected top 20 features and computed average importance scores.

### ⚙️ Feature Engineering:
- Created three new features:
  - `radius_ratio = radius_worst / radius_mean`
  - `area_ratio = area_worst / area_mean`
  - `concavity_product = concavity_worst * concavity_mean`
- Visualized score distributions and selected top common features across methods.

### 🧼 Preprocessing:
- Removed outliers using IQR.
- Final dataset `cl_data.csv` created with top features + engineered features.

---

## 🤖 2. Logistic Regression Model
**Notebook:** `Model_LG_Alhassan_DASI_305_Phase3.ipynb`

### 🧠 Model:
- Used `LogisticRegression(max_iter=1000)`
- Trained on top features selected during EDA.
- Evaluated using Accuracy, Confusion Matrix, and AUC-ROC.

### 🔎 XAI Techniques Used:
- ✅ **PDP** – Partial Dependence for top 10 features using `sklearn.inspection`
- ✅ **ICE** – Individual Conditional Expectation
- ✅ **ALE** – Using `alibi.explainers.ALE` on top features
- ✅ **SHAP** – Global & local explanations using `shap.Explainer` for `predict_proba`

### 📌 Findings:
- Most influential features: `area_worst`, `concavity_worst`, `area_ratio`, `perimeter_worst`.
- Logistic Regression offered clear interpretability and smooth SHAP outputs.
- ICE showed consistent patterns; ALE highlighted marginal effects.

---

## 🌲 3. Random Forest Model
**Notebook:** `Model_RF_Alhassan_DASI_305_Phase3.ipynb`

### 🧠 Model:
- Used `RandomForestClassifier(n_estimators=100)`
- Trained on same cleaned and engineered dataset.

### 🔎 XAI Techniques Used:
- ✅ **PDP** – for top 10 features
- ✅ **ICE** – with `sklearn.inspection` for individual instances
- ✅ **ALE** – with `alibi.explainers.ALE`

### 📌 Findings:
- RF performed slightly better on accuracy but had more complex decision boundaries.
- SHAP showed `area_worst`, `perimeter_worst`, and `concavity_product` as strong influencers.
- PDP and ALE captured nonlinear feature relationships, especially for worst-case metrics.

---

## 📁 Output Files:
- `cl_data.csv`: Cleaned and engineered feature dataset.
- `Model_LG_*.ipynb`: Logistic Regression + XAI
- `Model_RF_*.ipynb`: Random Forest + XAI
- `EDA_FeatureEngineering_*.ipynb`: Data preparation and feature pipeline

---

## 🧠 Conclusion:
- Both models provide strong performance and interpretability.
- Logistic Regression is ideal for clarity; Random Forest is better for capturing interactions.
- Combining multiple XAI tools gave the best insight into model behavior.
