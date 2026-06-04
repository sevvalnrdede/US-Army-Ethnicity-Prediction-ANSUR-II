# 🪖 US Army Soldier Ethnicity Prediction (ANSUR II Dataset)

An end-to-end Supervised Machine Learning project focused on classifying the ethnicity (`DODRace`) of U.S. Army personnel based on comprehensive body measurement indicators. This repository implements multiple statistical classifiers (**Logistic Regression, Support Vector Machines, Decision Trees, and XGBoost**) utilizing advanced feature scaling and imbalanced data optimization pipelines.

---

## 📌 Project Overview
Anthropometric datasets often feature high-dimensional geometric body measurements that exhibit underlying structural patterns tied to demographic backgrounds. This project builds a complete predictive pipeline using the benchmark **2012 Anthropometric Survey of U.S. Army Personnel (ANSUR II)** dataset. The main goal is to successfully classify soldiers into their respective ethnicities (`Black`, `Hispanic`, or `White`) using body dimensions while systematically handling heavy class imbalance.

---

## 📊 Dataset Specifications
* **Dataset Reference Name:** ANSUR II Databases (2012 Anthropometric Survey of U.S. Army Personnel)
* **Data Sources:** Combined dynamic tracking sheets for:
  * Male Soldiers: [Data World Repository](https://query.data.world/s/h3pbhckz5ck4rc7qmt2wlknlnn7esr)
  * Female Soldiers: [Data World Repository](https://query.data.world/s/sq27zz4hawg32yfxksqwijxmpwmynq)
* **Scope:** Dynamic physiological feature columns mapping dimensions such as `abdominalextensiondepthsitting`, `acromialheight`, `biacromialbreadth`, and `bicepscircumferenceflexed`.
* **Target Variable:** `DODRace` (Class Labels filtered for: `Black`, `Hispanic`, `White`).

---

## 🛠️ Technology Stack & Dependencies
* **Programming Language:** Python
* **Data Handling:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning Frameworks:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE/Class Weighting)

---

## 🚀 Machine Learning Implementation Pipeline

### 1. Exploratory Data Analysis (EDA) & Data Cleaning
* Merged independent male and female records into a centralized framework.
* Filtered low-representation minority demographic categories containing fewer than 50 data records.
* Discarded irrelevant spatial structural identifiers (`SubjectNumericID`, `WritingPreference`, etc.).

### 2. Preprocessing & Imbalanced Data Handling
* Set apart strict training and validation splits (`1154` standalone evaluation testing matrix).
* Executed numerical normalization standardization transformations (`MinMaxScaler`).
* Implemented algorithmic class optimization via customized loss weighting constraints (`Class Weights`) and sample resampling techniques to correct the heavy minority class under-representation (`Hispanic` classification limits).

### 3. Machine Learning Modeling & Hyperparameter Tuning
Implemented and optimized multiple multi-class classifier architectures through 10-Fold Cross-Validation (`GridSearchCV`):
* **Logistic Regression:** Tuned L2 regularization boundaries (`C=0.8`).
* **Support Vector Classifier (SVC):** Tuned structural scaling bounds (`C=0.5`, `gamma='scale'`).
* **Decision Tree Frameworks:** Established basic deterministic pruning partitions.
* **XGBoost Classifier:** Evaluated state-of-the-art gradient boosted trees optimized via specialized depth restrictions.

---

## 📈 Model Performance & Comparisons

The performance metrics collected from the comprehensive evaluation test set are summarized below:

| Classification Model | Pipeline Adjustments | Test Accuracy | Macro F1-Score | Performance Status |
| :--- | :--- | :---: | :---: | :---: |
| 🥇 **XGBoost Classifier** | Optimized Multi-Trees | **89.00%** | **0.84** | **Best Classifier** |
| 🥈 **Logistic Regression** | Balanced Base Weighting | 86.00% | 0.81 | Highly Interpretable |
| 🥉 **Support Vector Machine (SVM)**| Scaled RBF Kernel | 84.00% | 0.80 | Stable Border |
| 🎗️ **Decision Tree** | Unpruned Tree Model | 82.00% | 0.59 | Suffered Overfitting |

### 🔍 Crucial Analytical Insights
* **The Minority Class Challenge:** Without class optimization weighting metrics, early tree models completely missed the `Hispanic` label (yielding an F1-score of only `0.08`). Introducing advanced resampling techniques boosted the Hispanic recall dramatically (achieving up to `0.72` - `0.82` across models).
* **Boosting Dominance:** The **XGBoost framework outpaced all individual linear algorithms, securing a top test accuracy score of 89.00%**, perfectly balancing precision across both majority and minority ethnic sub-classes.

---

## 📁 Repository Structure
```text
├── notebooks/
│   └── ML_Supervised_Final_Project (Soldier Race).ipynb
├── .gitignore
└── README.md
