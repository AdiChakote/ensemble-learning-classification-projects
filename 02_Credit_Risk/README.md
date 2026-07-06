# Credit Default Risk Prediction using Machine Learning

## Project Overview

This project predicts whether a customer is likely to experience **serious credit delinquency (default) within the next two years** using historical financial and credit-related information.

The project follows an end-to-end machine learning classification workflow including data preprocessing, Exploratory Data Analysis (EDA), Weight of Evidence (WOE) Encoding, Information Value (IV), Variance Inflation Factor (VIF), feature selection, multiple machine learning models, hyperparameter tuning using GridSearchCV, and model comparison.

---

## Problem Statement

Financial institutions need to identify customers who are at high risk of default before approving loans or extending credit. Early prediction of credit default helps reduce financial losses and improves risk management.

The objective of this project is to classify customers into:

- **0 → No Default**
- **1 → Default**

---

## Dataset Information

- **Dataset:** Give Me Some Credit
- **Source:** Kaggle
- **Rows:** 150,000
- **Features:** 10
- **Target Variable:** `SeriousDlqin2yrs`

### Target Classes

| Value | Class      |
| ----- | ---------- |
| 0     | No Default |
| 1     | Default    |

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost Library

---

## Exploratory Data Analysis (EDA)

Performed:

- Dataset Understanding
- Data Cleaning
- Missing Value Analysis
- Missing Value Treatment
- Duplicate Value Check
- Descriptive Statistics
- Distribution Analysis
- Correlation Analysis
- Outlier Detection
- Target Variable Analysis

---

## Data Preprocessing

- Removed unnecessary index column
- Filled missing values using Median Imputation
- Removed weak predictor variables using Information Value (IV)
- Checked multicollinearity using Variance Inflation Factor (VIF)
- Feature and target separation

---

## Feature Engineering

### Weight of Evidence (WOE)

Applied WOE Encoding to convert predictor variables into statistically meaningful numerical values for credit risk modeling.

### Information Value (IV)

Calculated Information Value to evaluate the predictive strength of each feature.

### Variance Inflation Factor (VIF)

Performed VIF analysis to identify and remove multicollinearity among predictor variables.

---

## Machine Learning Models

The following models were trained using **GridSearchCV**:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     93.56% |     55.68% |     18.00% |     27.20% |
| Decision Tree       |     93.49% |     54.07% |     17.24% |     26.14% |
| Random Forest       |     93.28% |     49.31% |     18.44% |     26.84% |
| **AdaBoost**        | **93.52%** |     54.14% | **20.07%** | **29.29%** |
| XGBoost             | **93.57%** | **56.02%** |     17.64% |     26.83% |

---

## Best Model

**AdaBoost Classifier**

### Reason

- Highest Recall among all models.
- Highest F1 Score.
- Better balance between Precision and Recall.
- Most effective at identifying customers likely to default.

---

## Key Insights

- Customers with frequent late payments showed a significantly higher probability of default.
- Monthly Income was one of the strongest predictors of credit risk.
- Higher Debt Ratio increased the likelihood of default.
- Credit utilization was an important indicator of repayment behavior.
- Due to class imbalance, Precision, Recall, and F1 Score were more informative than Accuracy alone.

---

## Project Structure

```text
Credit_Risk/
│
├── Data/
│   ├── cs-training.csv
│
├── Notebook/
│   └── credit_risk.ipynb
│
├── README.md
```

---

## Future Improvements

- Handle class imbalance using SMOTE.
- Optimize the classification threshold.
- Experiment with additional ensemble learning algorithms.
- Deploy the model using Flask or FastAPI.
- Build an interactive dashboard using Streamlit.

---

## Author

**Aditya Sunil Chakote**

GitHub: https://github.com/AdiChakote
