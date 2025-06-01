import pandas as pd
df=pd.read_excel('data.xlsx', sheet_name="1")
print(df.head())
"""corr=df.corr()
print(corr)"""
X=df.drop(['Y1','Y2'], axis=1)
Y=df['Y1']
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
mse = mean_squared_error(Y_test, Y_pred)
rmse=np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

#cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validated RMSE: {cv_rmse.mean()} ± {cv_rmse.std()}")
cv_scores = cross_val_score(model, X, Y,  cv=5, scoring='r2')
print("CV R2 Score:", cv_scores.mean())
# losso and ridge regression
from sklearn.linear_model import Lasso, Ridge
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=0.1)
lasso.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
lasso_pred = lasso.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_mse = mean_squared_error(Y_test, lasso_pred)
ridge_mse = mean_squared_error(Y_test, ridge_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(Y_test, lasso_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(Y_test, ridge_pred)
print(f"Lasso RMSE: {lasso_rmse}")
print(f"Lasso R-squared: {lasso_r2}")
print(f"Ridge RMSE: {ridge_rmse}")
print(f"Ridge R-squared: {ridge_r2}")    

X = df.drop(['Y1', 'Y2'], axis=1)
Y = df['Y1']
#Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split for Y1
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
# Train the model
model2 = LinearRegression()
model2.fit(X_train, Y_train)
# Predict on the test set
importance2 = pd.Series(model2.coef_, index=X.columns).sort_values(ascending=False)
print("Feature Importance after Scaling:")
print(importance2)
# Feature importance
importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(importance)

XF1 = df.drop(['Y1', 'Y2','X3','X8','X6'], axis=1)
YF1 = df['Y1']
# Train-test split for reduced features
X_trainF1, X_testF1, Y_trainF1, Y_testF1 = train_test_split(XF1, YF1, test_size=0.2, random_state=42)
# Train the model with reduced features
modelF1 = LinearRegression()
modelF1.fit(X_trainF1, Y_trainF1)
# Predict on the test set with reduced features
YF1_pred = modelF1.predict(X_testF1)
# Evaluate the model with reduced features
mseF1 = mean_squared_error(Y_testF1, YF1_pred)
rmseF1 = np.sqrt(mseF1)
r2F1 = r2_score(Y_testF1, YF1_pred)
print(f"Reduced Features1 - Root Mean Squared Error: {rmseF1}")
print(f"Reduced Features1 - R-squared: {r2F1}")

XF2 = df.drop(['Y1', 'Y2','X6','X8'], axis=1)
YF2 = df['Y1']
# Train-test split for reduced features
X_trainF2, X_testF2, Y_trainF2, Y_testF2 = train_test_split(XF2, YF2, test_size=0.2, random_state=42)
# Train the model with reduced features
modelF2 = LinearRegression()
modelF2.fit(X_trainF2, Y_trainF2)
# Predict on the test set with reduced features
YF2_pred = modelF2.predict(X_testF2)
# Evaluate the model with reduced features
mseF2 = mean_squared_error(Y_testF2, YF2_pred)
rmseF2 = np.sqrt(mseF2)
r2F2 = r2_score(Y_testF2, YF2_pred)
print(f"Reduced Features2 - Root Mean Squared Error: {rmseF2}")
print(f"Reduced Features2 - R-squared: {r2F2}")

cvF2_scores = cross_val_score(model, XF2, YF2, cv=5, scoring='neg_mean_squared_error')
cvF2_rmse = np.sqrt(-cvF2_scores)
print(f"Cross-validated RMSE: {cvF2_rmse.mean()} ± {cv_rmse.std()}")
cvF2_scores = cross_val_score(model, X, Y,  cv=5, scoring='r2')
print("CV R2 Score:", cvF2_scores.mean())