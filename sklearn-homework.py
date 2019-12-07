import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import load_diabetes
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, max_error

def description(ds):
    print(f'Shape of dataset : {ds.shape}')
    print(f'stats of dataset: \n {ds.describe()}')

def performance(title, reg_model, X,y, y_test, predicted_values, columns_names):
    print(title)
    if title == "Linear Regression Performance":
        print(f"Named Coeficients: {pd.DataFrame(reg_model.coef_, columns_names)}")
    print("Mean squared error ( close to 0 the better): %.2f"
          % mean_squared_error(y_test, predicted_values))
    print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))
    print('score ( close to 1.0 the better): %.2f' % reg_model.score(X, y))
    if title == "Linear Regression Performance":
        print("Intercept: %.2f" % reg_model.intercept_)
    print("Max Error ( close to 0.0 the better): %.2f" % max_error(y_test, predicted_values))



diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

predicted_values = lm.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

residuals = y_test - predicted_values

# Performance Info
from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))


# Using KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
# we will process a loop to find the best performance for KNN for max_number_of_neighbors
max_number_of_neighbors = 50
neighbor = 1
min_mean_sqr_error = 0
max_r2_score = 0
opt_neighbor = 0
global optimized_neighbor
while neighbor < max_number_of_neighbors:
    model = KNeighborsRegressor()
    model.n_neighbors = neighbor
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    mean_sqr_error = mean_squared_error(y_test, predicted_values)
    r2_score_calc = r2_score(y_test, predicted_values)
    print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
    print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
    print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
    print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))
    neighbor = neighbor + 1
