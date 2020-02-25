#Q3: Does extremist vote share in parliament has any impact on number of protests and real GDP of Germany and USA?

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
orig_dataset = orig_dataset.interpolate(method='linear',limit_area ='inside')
orig_dataset = orig_dataset.fillna(0)

# Regions
Germany = ['DEU']
America = ['USA']

# Dataset division in different regions
germany = orig_dataset[orig_dataset.iso.isin(Germany)]
america = orig_dataset[orig_dataset.iso.isin(America)]

# Select relevant features from data
feature_cols=['protests', 'protestsdev', 'demosdev', 'riotsdev', 'strikesdev', 'rgdp']

                                                            # european nations model
X = orig_dataset[feature_cols]
y = orig_dataset.extr


# performing recursive feature selection for best features to test different models.
from sklearn.tree import DecisionTreeRegressor
cols = feature_cols
model = DecisionTreeRegressor()
#Initializing RFE model
rfe = RFE(model, 4)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


print('impact on european nations')
#Now use models to predict on selected features
feature_cols_new = ['demosdev', 'riotsdev', 'strikesdev', 'rgdp']
X = germany[feature_cols_new]
y = germany.extr

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

#build multiple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('Multiple Linear Regression Results For Germany')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 on train data ::',r2_score(y_train, y_pred_train, multioutput='variance_weighted'))
print('R2 on test data::',r2_score(y_test, y_pred, multioutput='variance_weighted'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = rf_regressor.predict(X_train)
y_pred = rf_regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('Random Forest Regressor Results For Germany')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 on train data with RFR ::',r2_score(y_train, y_pred_train, multioutput='variance_weighted'))
print('R2 on test data RFR::',r2_score(y_test, y_pred, multioutput='variance_weighted'))


# Regularization

# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(regressor.coef_, predictors).sort_values()
plt.rcParams['figure.figsize'] = [10, 10] # for square canvas
plt.rcParams['figure.subplot.bottom'] = 0.15
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()


# Impact on American nations
print('Impact on American Nations')

feature_cols_new = ['demosdev', 'riotsdev', 'strikesdev', 'rgdp']
X = america[feature_cols_new]
y = america.extr

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('Multiple Linear Regression Results For USA')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 on train data ::',r2_score(y_train, y_pred_train, multioutput='variance_weighted'))
print('R2 on test data::',r2_score(y_test, y_pred, multioutput='variance_weighted'))

#build multiple linear regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = rf_regressor.predict(X_train)
y_pred = rf_regressor.predict(X_test)

# Estimating this model
from sklearn import metrics
print('Random Forest Regression Results for USA')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 on train data ::',r2_score(y_train, y_pred_train, multioutput='variance_weighted'))
print('R2 on test data ::',r2_score(y_test, y_pred, multioutput='variance_weighted'))

# Regularization

# calculating coefficients
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,y_train)
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(linear_regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(linear_regressor.coef_, predictors).sort_values()
plt.rcParams['figure.figsize'] = [10, 10] # for square canvas
plt.rcParams['figure.subplot.bottom'] = 0.15
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

