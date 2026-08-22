# Credit Card Default Prediction

A machine learning project that predicts whether a credit card client will default on their payment next month, based on demographic information, credit data, and past payment history.

## 📌 Objective

Credit card default prediction is a key problem for banks and financial institutions. Predicting default risk in advance helps them:
- Assess credit risk before issuing or renewing credit
- Set appropriate credit limits
- Flag high-risk clients early to reduce financial losses

This project builds a **binary classification model** to predict the target variable `default payment next month` (1 = default, 0 = no default).

## 📊 Data Source

This project uses the **Default of Credit Card Clients** dataset, originally published by I-Cheng Yeh, hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

The dataset contains information on **30,000 credit card clients in Taiwan** (April–September 2005), including:
- **Demographics:** `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Credit data:** `LIMIT_BAL` (amount of given credit)
- **Repayment history:** `PAY_0` to `PAY_6`
- **Bill statement amounts:** `BILL_AMT1` to `BILL_AMT6`
- **Previous payment amounts:** `PAY_AMT1` to `PAY_AMT6`
- **Target variable:** `default payment next month`

## 🛠️ Tools & Libraries

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## 🔍 Project Workflow

1. **Data Import & Cleaning** — loaded the dataset, checked for missing values/duplicates, and cleaned undocumented category codes in `EDUCATION` and `MARRIAGE`.
2. **Exploratory Data Analysis** — visualized target class distribution, correlation heatmap, age distribution, and credit limit by default status.
3. **Data Preprocessing** — feature scaling with `StandardScaler` for logistic regression.
4. **Train/Test Split** — 75/25 split, stratified on the target to preserve class balance.
5. **Modeling** — trained and compared two models:
   - Logistic Regression (baseline, interpretable)
   - Random Forest Classifier (non-linear, stronger performance)
6. **Model Evaluation** — accuracy, confusion matrix, precision/recall/F1-score for both models.
7. **Feature Importance** — identified the most influential predictors using the Random Forest model.
8. **Prediction** — demonstrated prediction on a sample client from the test set.

## 📈 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~81.6% |
| Random Forest | ~82.9% |

Random Forest outperformed Logistic Regression, particularly in recall/F1-score for identifying actual defaulters. The most important predictor was `PAY_0` (most recent repayment status), followed by other recent repayment status features and `LIMIT_BAL` — confirming that recent payment behavior is the strongest signal of future default.

## 📁 Files

- `Credit_Card_Default_Prediction.ipynb` — full Jupyter notebook with code, visualizations, and results
- `credit_card_default.csv` — cleaned dataset used in the notebook

## 🚀 How to Run

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Open Credit_Card_Default_Prediction.ipynb in Jupyter Notebook, JupyterLab, VS Code, or Google Colab
5. Run all cells (make sure credit_card_default.csv is in the same folder)
