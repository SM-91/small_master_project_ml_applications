# Q: Does the number of protests have any impact on RGDP among different regions?
# Q: Does the political instability have any impact on RGDP among different regions?


#Importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Dataset Display Settings

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 5000)


# importing dataset
orig_dataset = pd.read_csv(r'C:\Users\Shayan Mahmood\Desktop\small Master project\Dataset.csv', sep= ',', header=0)

# Handling missing data
orig_dataset = orig_dataset.interpolate(method='linear',limit_area ='inside')
orig_dataset = orig_dataset.fillna(0)


# Regions
Europe = ['AUT','BEL','CHE','DEU','DNK','ESP','FIN','FRA','GBR','GRC','IRL','ITA','NLD','NOR','PRT','SWE']
North_America = ['USA','CAN']
Eastern_block = ['AUS','JPN']

# Dataset division in different regions
european_nations = orig_dataset[orig_dataset.iso.isin(Europe)]
american_nations = orig_dataset[orig_dataset.iso.isin(North_America)]
eatern_nations = orig_dataset[orig_dataset.iso.isin(Eastern_block)]


# Select relevant features from data
                                               # european nations model
feature_cols=['govvote', 'oppvote', 'frac', 'partycount', 'right', 'left', 'extr', 'protests', 'protestsdev', 'demosdev', 'riotsdev', 'strikesdev','turnover','govcris','vetopl']
X = orig_dataset[feature_cols]
y = orig_dataset.rgdp

# Feature Selection by embedded method
from sklearn.linear_model import LassoCV
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()

feature_cols_new = ['partycount','left','extr','govvote','protests']
X = european_nations[feature_cols_new]
y = european_nations.rgdp

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Applying Multiple Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

print('Multiple Regression Results in European Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('European nations results')
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train  ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test  ::',r2_score(y_test, y_pred, multioutput='uniform_average'))

# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor()
random.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = random.predict(X_train)
y_pred = random.predict(X_test)

print('Random Forest Regressor Results in European Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train  ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test  ::',r2_score(y_test, y_pred, multioutput='uniform_average'))

                                                # model of American nations

feature_cols_new = ['partycount','left','extr','govvote','protests']
X = american_nations[feature_cols_new]
y = american_nations.rgdp

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Applying Multiple Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train= regressor.predict(X_train)
y_pred = regressor.predict(X_test)

print('Multiple Linear Regression Results in American Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('American Nations results')
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test ::',r2_score(y_test, y_pred, multioutput='uniform_average'))

# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor()
random.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = random.predict(X_train)
y_pred = random.predict(X_test)

print('Random Forest Regressor Results in American Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train  ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test  ::',r2_score(y_test, y_pred, multioutput='uniform_average'))

                                                            # Eastern Nation model

feature_cols_new = ['partycount','left','extr','govvote','protests']
X = eatern_nations[feature_cols_new]
y = eatern_nations.rgdp

# Splitting the dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Applying Multiple Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(X_test)


print('Multiple Linear Regression Results in Eastern Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('Eastern Nations Results')
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test ::',r2_score(y_test, y_pred, multioutput='uniform_average'))

# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(regressor.coef_)
print(coeff)

predictors = X_train.columns
coef = pd.Series(regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor()
random.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = random.predict(X_train)
y_pred = random.predict(X_test)

print('Random Forest Regressor Results in Eastern Nations')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE ::',mean_absolute_error(y_test, y_pred))
print('MSE ::',mean_squared_error(y_test, y_pred))
print('R2 Train  ::',r2_score(y_train, y_pred_train, multioutput='uniform_average'))
print('R2 Test  ::',r2_score(y_test, y_pred, multioutput='uniform_average'))