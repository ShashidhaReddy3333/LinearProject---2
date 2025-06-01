# 🔥 Energy Efficiency Prediction with Regularized Linear Regression

## 🧠 Project Summary

This project predicts energy efficiency (heating load) using a regularized linear regression model. It uses the UCI Energy Efficiency dataset and evaluates multiple regression techniques including Ridge and Lasso. The goal is to explore how different architectural features influence heating demand.

---

## 📂 Dataset

* **Source:** UCI Machine Learning Repository
* **File:** `data.xlsx`
* **Sheet Name:** `1`
* **Target Variable:** `Y1` (Heating Load)

---

## 📌 Features Used

| Feature | Description                     |
| ------- | ------------------------------- |
| X1      | Relative Compactness            |
| X2      | Surface Area (m²)               |
| X3      | Wall Area (m²)                  |
| X4      | Roof Area (m²)                  |
| X5      | Overall Height (m)              |
| X6      | Orientation (0-4)               |
| X7      | Glazing Area (%)                |
| X8      | Glazing Area Distribution (0-5) |

---

## ⚙️ Model Workflow

1. **Data Loading** from Excel
2. **Train-Test Split** (80-20)
3. **Baseline Linear Regression**
4. **Cross-validation (5-fold)**
5. **Regularization:** Ridge and Lasso
6. **Standardization:** Feature Scaling using `StandardScaler`
7. **Feature Importance** before and after scaling

---

## 🧪 Model Performance

| Model             | RMSE          | R²      |
| ----------------- | ------------- | ------- |
| Linear Regression | 3.025         | 0.912   |
| Cross-Validated   | 2.941 ± 0.116 | 0.914   |
| Lasso Regression  | \~3.17        | \~0.903 |
| Ridge Regression  | \~3.06        | \~0.909 |

✅ The Linear Model explains \~91% of the variance with an RMSE under 10% of the target range (max Y ≈ 43).

---

## 📊 Key Findings

* **Most influential features:**

  * 📉 X1 (Relative Compactness) – Strong negative correlation
  * 📉 X4, X2 (Roof & Surface Area) – Negative impact
  * 📈 X5 (Overall Height) – Strong positive effect
  * 📈 X7 (Glazing Area) – Positive effect

* **After Scaling:** X5, X1, and X4 show the largest absolute coefficients.

---

## 📁 Project Structure

```
LinearProject - 2/
├── data.xlsx
├── energy_regression.py
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# Create virtual environment
python -m venv .venv
. .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python energy_regression.py --data data.xlsx --sheet 1 --target Y1
```

---

## 📌 Notes

* The code is modular and organized for easy experimentation.
* You can change the target to `Y2` (Cooling Load) using `--target Y2`.
* Compatible with Python 3.12 and scikit-learn ≥ 1.3

---

## 📈 Future Improvements

* Polynomial features and interaction terms
* Grid search for optimal alpha in Ridge/Lasso
* Try other regressors (SVR, Gradient Boosting)

---

## 📜 License

MIT License
