import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv("C:/Users/jonna/Downloads/p2_test.csv")
test_data = pd.read_csv("C:/Users/jonna/Downloads/p2_test.csv")

# Separate features and target variable
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Regression (SVR)
svr = SVR(kernel='rbf')  # Using radial basis function kernel for non-linearity
svr.fit(X_train, y_train)
svr_predictions = svr.predict(X_test)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_predictions = linear_reg.predict(X_test)


# Evaluation
def evaluate_model(predictions, y_true):
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    return mse, mae


svr_mse, svr_mae = evaluate_model(svr_predictions, y_test)
linear_mse, linear_mae = evaluate_model(linear_predictions, y_test)

print("Support Vector Regression:")
print(f"Mean Squared Error (SVR): {svr_mse}")
print(f"Mean Absolute Error (SVR): {svr_mae}\n")

print("Linear Regression:")
print(f"Mean Squared Error (Linear Regression): {linear_mse}")
print(f"Mean Absolute Error (Linear Regression): {linear_mae}")

# Visualize results (SVR vs Linear Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, svr_predictions, label='SVR Predictions')
plt.scatter(y_test, linear_predictions, label='Linear Regression Predictions')
plt.xlabel('Actual Lifespan')
plt.ylabel('Predicted Lifespan')
plt.title('SVR vs Linear Regression Predictions')
plt.legend()
plt.show()
