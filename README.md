# Loan Approval Prediction

> **Banks reject good applicants and approve bad ones — not always because the data is wrong, but because the process is. This project replaces guesswork with a machine learning pipeline that predicts loan approval outcomes with near-perfect accuracy.**

---

## Problem Statement

Access to credit is a critical financial need, yet loan approval processes are often slow, inconsistent, and prone to human bias. Financial institutions must evaluate hundreds of applications, balancing the risk of approving bad loans against the cost of rejecting creditworthy applicants. Manual review is neither scalable nor reliable at volume.

This project addresses that challenge by building a machine learning pipeline that automates loan approval decisions. Given an applicant's financial profile, the model predicts whether a loan application should be approved or rejected — providing a fast, data-driven, and reproducible decision process.

---

## Solution Overview

Two classification models — Logistic Regression and Decision Tree — were trained and evaluated on a real-world loan dataset. Class imbalance was addressed using SMOTE to ensure the models do not overlook rejected applications. The final system achieves near-perfect precision and recall, demonstrating that automated credit screening is both feasible and reliable with the right approach.

---

## Dataset

- **Source:** [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) via KaggleHub
- **Size:** 4,269 records
- **Target:** `loan_status` — Approved (62%) / Rejected (38%)
- **Features:** Annual income, loan amount, loan term, CIBIL score, residential assets, commercial assets, luxury assets, bank assets, education level, self-employment status

---

## Project Structure

```
Loan_Approval_Prediction.ipynb
│
├── Libraries
├── Dataset
├── Exploratory Data Analysis
├── Preprocessing Data
├── Logistic Regression Model
├── SMOTE (Class Imbalance Handling)
├── Decision Tree
└── Logistic Regression vs Decision Tree
```

---

## Exploratory Data Analysis

- The dataset has a moderate class imbalance — 62% approved, 38% rejected — so evaluation focuses on precision, recall, and F1-score rather than accuracy alone.
- **CIBIL score** (India's credit scoring system, ranging from 300 to 900) shows the clearest separation between approved and rejected loans and is the strongest individual predictor.
- Several asset-related features contain outliers, detected using the IQR method.
- Categorical features (`education`, `self_employed`) were visualized against loan status to assess their influence on approval outcomes.

---

## Preprocessing

- Stripped whitespace from column names to prevent mapping errors during label encoding.
- Label-encoded categorical features (`education`, `self_employed`) and the target variable (`loan_status`) into binary integers.
- Applied `StandardScaler` on numerical features after the train/test split to prevent data leakage — the scaler is fit only on training data.
- Applied **SMOTE** on the training set to handle class imbalance before model training.

---

## Models

### Logistic Regression

Logistic Regression is a linear classification algorithm that models the probability of a binary outcome using a logistic (sigmoid) function. It estimates the relationship between each input feature and the log-odds of the target class, making it interpretable and computationally efficient. It works well when the relationship between features and the target is approximately linear, and serves here as a strong, interpretable baseline.

### SMOTE — Synthetic Minority Oversampling Technique

The dataset contains more approved loans than rejected ones. If left unaddressed, models tend to favour the majority class and perform poorly at identifying rejections — which is the more costly error in a credit context. SMOTE addresses this by synthetically generating new samples for the minority class (Rejected) by interpolating between existing minority examples in feature space, rather than simply duplicating records. This produces a more balanced training set without discarding majority class data. SMOTE is applied only to the training set to avoid leaking synthetic data into evaluation.

### Decision Tree

A Decision Tree is a non-linear model that partitions the feature space into regions by learning a hierarchy of binary rules (e.g., "CIBIL score > 650"). Each internal node represents a decision on a feature, each branch represents an outcome of that decision, and each leaf node represents a predicted class. Decision Trees can capture complex, non-linear interactions between features without requiring feature scaling, and are highly interpretable — the learned rules can be visualised and explained directly.

---

## Results

### Logistic Regression (Original)

| Loan Status | Precision | Recall | F1-score |
|---|---|---|---|
| Rejected | 0.90 | 0.86 | 0.88 |
| Approved | 0.92 | 0.94 | 0.93 |

### Logistic Regression + SMOTE

Applying SMOTE before training improved recall for the minority class (Rejected loans), reducing the number of rejected applications that were incorrectly classified as approved — the more consequential error in a lending context.

### Decision Tree

| Loan Status | Precision | Recall | F1-score |
|---|---|---|---|
| Rejected | ~0.98 | ~0.99 | ~0.98 |
| Approved | ~0.99 | ~0.98 | ~0.99 |

---

### Final Comparison

| Model | Performance |
|---|---|
| Logistic Regression (Original) | Good overall, slightly lower recall for rejected loans |
| Logistic Regression + SMOTE | Improved minority class recall |
| **Decision Tree** | **Best — near-perfect precision and recall across both classes** |

The Decision Tree consistently outperformed Logistic Regression across all metrics, achieving near-perfect results without requiring SMOTE. Its ability to model non-linear feature interactions — particularly around CIBIL score thresholds — is likely the primary reason for this performance advantage.

---

## Libraries Used

```python
numpy, pandas, matplotlib, seaborn
scikit-learn (LogisticRegression, DecisionTreeClassifier, StandardScaler)
imbalanced-learn (SMOTE)
kagglehub
```

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Moamen-Elgohary/Loan-Approval-Prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install kaggle kagglehub scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
   ```
3. Open and run the notebook — the dataset will be downloaded automatically via KaggleHub:
   ```bash
   jupyter notebook Loan_Approval_Prediction.ipynb
   ```
   Alternatively, the notebook can be run directly in **Google Colab**.

---

## License

The dataset used in this project is made available under the [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license, meaning it can be used, modified, and distributed freely without restriction.

This project was completed as part of the Elevvo Pathways internship program.
