# Q8: Does far left and far right vote share have any impact on consumer price index?

#Importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Dataset Display Settings
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 5000)

# importing dataset
orig_dataset = pd.read_csv(r'C:\Users\Shayan Mahmood\Desktop\small Master project\Dataset.csv', sep= ',', header=0)
print(orig_dataset.columns)

# Handling missing data
orig_dataset = orig_dataset.interpolate(method='linear',limit_area ='inside')
orig_dataset = orig_dataset.fillna(0)
print(orig_dataset.get_dtype_counts())

# Select relevant features from data
feature_cols=['right', 'left']

X = orig_dataset[feature_cols]
y = orig_dataset.cpi

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


lr_regressor = LinearRegression()
# MSEs = cross_val_score(lr_regressor,X,y, scoring= 'neg_mean_squared_error', cv =10)
# mean_MSE = np.mean(MSEs)
# print('Mean Square Error::',mean_MSE)
lr_regressor.fit(X_train, y_train)

# Predicting the Train and Test set values
y_pred_train_lr = lr_regressor.predict(X_train)
y_pred_test_lr = lr_regressor.predict(X_test)

#Predicting R2 for train and test
print('R2 Score of train set::',r2_score(y_train,y_pred_train_lr))
print('R2 score of test set::',r2_score(y_test,y_pred_test_lr))
print('Mean Absolute Error with LR ::',mean_absolute_error(y_test, y_pred_test_lr))

# Regularization
# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(lr_regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(lr_regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()


from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor(criterion='mae', splitter='best')
dt_regressor.fit(X_train,y_train)
# Predicting the Train and Test set values
y_pred_train_dt = dt_regressor.predict(X_train)
y_pred_test_dt = dt_regressor.predict(X_test)

#Predicting R2 for train and test
print('R2 Score of train set::',r2_score(y_train,y_pred_train_dt))
print('R2 score of test set::',r2_score(y_test,y_pred_test_dt))
print('Mean Absolute Error with DTR ::',mean_absolute_error(y_test, y_pred_test_dt))


# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge
# ridge = Ridge()
# ridge.fit(X_train,y_train)
# # parameters = {'alpha' : [1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20]}
# # ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
# # ridge_regressor.fit(X_train,y_train)
# y_pred_train_ridge = ridge.predict(X_train)
# y_pred_test_ridge = ridge.predict(X_test)
#
# #Predicting R2 for train and test
# print('R2 Score of train set::',r2_score(y_train,y_pred_train_ridge))
# print('R2 score of test set::',r2_score(y_test,y_pred_test_ridge))
# print('Mean Absolute Error with Ridge ::',mean_absolute_error(y_test, y_pred_test_ridge))
#
# # print('Best Parameter from Ridge Regressor::', ridge_regressor.best_params_)
# # print('Best Score from Ridge Regressor::', ridge_regressor.best_score_)
# # print('Best estimator::',ridge_regressor.best_estimator_)
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Lasso
# lasso = Lasso()
# parameters = {'alpha' : [1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20]}
# lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=10)
# lasso_regressor.fit(X,y)
#
# print('Best Parameter from Lasso Regressor::', lasso_regressor.best_params_)
# print('Best Score from Lasso Regressor::', lasso_regressor.best_score_)
# print('Best estimator::',lasso_regressor.best_estimator_)












