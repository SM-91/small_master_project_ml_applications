# Q: how does partycount effect fractionalisation of parliament in western Europe?
#Importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Dataset Display Settings
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 5000)


# importing dataset
orig_dataset = pd.read_csv(r'C:\Users\Shayan Mahmood\Desktop\small Master project\Dataset.csv', sep= ',', header=0)
print(orig_dataset.columns)

# Handling missing data

orig_dataset = orig_dataset.interpolate(method='linear')
orig_dataset = orig_dataset.fillna(0)


# Regions
Europe = ['BEL','DEU','DNK','ESP','FRA','GBR','IRL','NLD','NOR','PRT','AUT','ITA']
North_America = ['USA','CAN']

# Dataset division in different regions
western_europe_nations = orig_dataset[orig_dataset.iso.isin(Europe)]
american_nations = orig_dataset[orig_dataset.iso.isin(North_America)]

# Select relevant features from data
feature_cols=['frac']

X = western_europe_nations[feature_cols]
y = western_europe_nations.partycount

plt.scatter(western_europe_nations['frac'], western_europe_nations['partycount'], color='red')
plt.title('frac Vs partycount', fontsize=14)
plt.xlabel('frac', fontsize=20)
plt.ylabel('cpi', fontsize=14)
plt.grid(True)
plt.show()

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

#build multiple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('Linear Regression Results')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred_test))
print('MSE ::',mean_squared_error(y_test, y_pred_test))
print('R2 with train set ::',r2_score(y_train, y_pred_train))
print('R2 with test set::',r2_score(y_test, y_pred_test))

# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

print('Ridge Results')
# Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
y_pred_train_ridge = ridge.predict(X_train)
y_pred_test_ridge = ridge.predict(X_test)

#Predicting R2 for train and test
print('R2 Score of train set::',r2_score(y_train,y_pred_train_ridge))
print('R2 score of test set::',r2_score(y_test,y_pred_test_ridge))
print('Mean Absolute Error with Ridge ::',mean_absolute_error(y_test, y_pred_test_ridge))
print('Mean Squared Error with Ridge ::',mean_squared_error(y_test, y_pred_test_ridge))


print('DT Results')
# Now using decision tree regressor
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_train_dt = dt_regressor.predict(X_train)
y_pred = dt_regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('DTR Results')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE with DTR ::',mean_absolute_error(y_test, y_pred))
print('MSE with DTR ::',mean_squared_error(y_test, y_pred))
print('R2 with DTR on train set ::',r2_score(y_train, y_pred_train_dt))
print('R2 with DTR on test set ::',r2_score(y_test, y_pred))

#parameter tuning

print('Results after parameter tuning')
parameters = {'alpha' : [1e-15,1e-10,1e-8,1e-4,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='r2',cv=10)
ridge_regressor.fit(X_train,y_train)
print('Best Parameter from Ridge Regressor::', ridge_regressor.best_params_)
print('Best Score from Ridge Regressor::', ridge_regressor.best_score_)
print('Best estimator::',ridge_regressor.best_estimator_)

print('Results of parameter tuning with DTR')
parameters_dt = {
    'criterion' : ['mae','mse'],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'splitter': ['best','random']}

dtr = GridSearchCV(dt_regressor,parameters_dt,scoring='r2',cv=10)
dtr.fit(X_train,y_train)
print('Best Parameter from DT Regressor::', dtr.best_params_)
print('Best Score from DT Regressor::', dtr.best_score_)
print('Best estimator::',dtr.best_estimator_)


#Model Evaluation after parameter tuning

ridge_new = Ridge(alpha=1)
ridge_new.fit(X_train,y_train)
y_pred_train_ridge_new = ridge_new.predict(X_train)
y_pred_test_ridge_new = ridge_new.predict(X_test)

#Predicting R2 for train and test
print('R2 Score of train set::',r2_score(y_train,y_pred_train_ridge_new))
print('R2 score of test set::',r2_score(y_test,y_pred_test_ridge_new))
print('Mean Absolute Error with Ridge ::',mean_absolute_error(y_test, y_pred_test_ridge_new))
print('Mean Squared Error with Ridge ::',mean_squared_error(y_test, y_pred_test_ridge_new))

dtr_new = DecisionTreeRegressor(criterion='mse',max_depth=10,splitter='best')
dtr_new.fit(X_train,y_train)
y_pred_train_dtr_new = dtr_new.predict(X_train)
y_pred_test_dtr_new = dtr_new.predict(X_test)

#Predicting R2 for train and test
print('R2 Score of train set::',r2_score(y_train,y_pred_train_dtr_new))
print('R2 score of test set::',r2_score(y_test,y_pred_test_dtr_new))
print('Mean Absolute Error with Ridge ::',mean_absolute_error(y_test, y_pred_test_dtr_new))
print('Mean Squared Error with Ridge ::',mean_squared_error(y_test, y_pred_test_dtr_new))