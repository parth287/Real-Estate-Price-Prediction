import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# For saving the model
from joblib import dump, load

housing = pd.read_csv("data.csv")
# print(housing.info())
# print(housing['CHAS'].value_counts())

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_size, test_size in split.split(housing, housing['CHAS']):
    train_set = housing.loc[train_size]
    test_set = housing.loc[test_size]

# print(len(train_set))
# print(train_set['CHAS'].value_counts())
# print(len(test_set))

# Assingning the training data to the actual data
housing_test = test_set.drop("MEDV", axis =1)
housing_test_label = test_set["MEDV"].copy()
housing_train = train_set.drop("MEDV",  axis = 1)
housing_train_label =train_set["MEDV"].copy() 
# Seperating the features and labes to pass it to the model

# Making the dataset more accurate by increasing the number of columns
housing_train["TAXRM"] = housing_train["TAX"]/housing_train["RM"]
housing_test["TAXRM"] = housing_test["TAX"]/housing_test["RM"]


# # Filling in the missing values
# #1 Removing the particular rows
# r = housing_train.dropna(subset = ["RM"])
# # print(r.shape)

# # 2 Removing the whole column that has missing values
# r = housing_train.drop("RM", axis = 1).shape

# 3 Replacing the blank columns with median which can fit in every missing place rather than RM too
# BEST OPTION
median = housing_train["RM"].median()
# Filling the missing values with the median values
housing_train["RM"].fillna(median) 

# Making a pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")), # step1
    ('std_scaler', StandardScaler()) 

])

# Using the imputer from the pipeline
pipe.fit(housing_train) 
x = pipe.transform(housing_train)
housing_train = pd.DataFrame(x , columns = housing_train.columns)
# print(housing_train.info())

# Using the standardized scaler from the pipeline
housing_train_tr = pipe.fit_transform(housing_train) #This a numpy array now, so no dataframe operations can be done on this
# print(housing_train_tr.shape) 

# Trying the Linear Regression MOdel
# model = LinearRegression()

# As the loss function is very high we'll use another model for the problem
# model = DecisionTreeRegressor()

# We'll try RandomForestRegressor algo to improve the inputs more
model = RandomForestRegressor()

model.fit(housing_train_tr, housing_train_label)
housing_pred = model.predict(housing_train_tr)
# Computing the loss function which is root of mean square error
lf_mse = mean_squared_error(housing_train_label, housing_pred)
lf_rmse = np.sqrt(lf_mse)
# print(lf_mse) # The ouput for Linear Regression is 23.390938364073325 

# The ouput for Decision Tree Regression is 0.0, the model is clearly overfitting

# To avoid overfitting we'll use Cross Validation   
# 1 2 3 4 5 6 7 8 9 10
scores = cross_val_score(model, housing_train_tr, housing_train_label,scoring="neg_mean_squared_error", cv = 10)
rmse = np.sqrt(-scores)
# print(rmse)
# scores for DTR are : 6.87997093 3.90861229 3.57498252 4.56442767
# scores for Linear REg are : [4.22542831 4.26475024 5.09807631 3.83086258 5.37290328 4.4212011
#  7.47087044 5.48909418 4.14340514 6.07123845]
# we choose DTR as it has less RSME
# print(housing_pred, list(housing_train_label))

# Printing The Scores, mean, STd Deviation
def printscores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard Deviation : ", scores.std())

# SAving the model

# dump(model, "RealEstate.joblib")

# Using the model via this function
def using_model(data,label):
    pipe.fit(data)
    x = pipe.transform(data)
    data = pd.DataFrame(x, columns = data.columns)
    data_tr = pipe.fit_transform(data)
    pred = model.predict(data_tr)
    f_mse = mean_squared_error(housing_test_label, pred)
    f_rsme = np.sqrt(f_mse)
    print("The mean square error is : ",f_mse)
    print("The root of mean square error is : ",f_rsme)
    print(pred, list(housing_test_label))
