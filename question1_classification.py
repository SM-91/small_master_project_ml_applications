# Q: How systematic banking crisis effect political stability among european nations?

#Importing Libs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


#Dataset Display Settings
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 5000)


# importing dataset
orig_dataset = pd.read_csv(r'C:\Users\Shayan Mahmood\Desktop\small Master project\Dataset.csv', sep= ',', header=0)

# Handling missing data
orig_dataset = orig_dataset.interpolate(method='linear',limit_area ='inside')
orig_dataset = orig_dataset.fillna(0)
print(orig_dataset.get_dtype_counts())

Europe = ['AUT','BEL','CHE','DEU','DNK','ESP','FIN','FRA','GBR','GRC','IRL','ITA','NLD','NOR','PRT','SWE']
european_nations = orig_dataset[orig_dataset.iso.isin(Europe)]

feature_cols=['govvote', 'oppvote', 'frac', 'partycount', 'right', 'left', 'extr', 'protests', 'protestsdev', 'demosdev', 'riotsdev', 'strikesdev', 'govcris', 'turnover', 'vetopl']
X = european_nations[feature_cols]
y = european_nations.crisisJST

#model selection using Train/Test splits
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Learning Models
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled,y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_pred_train = knn_model.predict(X_train_scaled)
# Perform cross validation on training set
knn_scores = cross_val_score(knn_model, X,y, cv=10)
knn_scores_mean = knn_scores.mean()
knn_scores_std = knn_scores.std()

# Compute the accuracy of the prediction
acc = float((y_test == knn_pred).sum()) / knn_pred.shape[0]
acc_train = float((y_train == knn_pred_train).sum()) / knn_pred_train.shape[0]

print('Test set accuracy in KNN: %.2f %%' % (acc * 100))
print('Train set accuracy in KNN: %.2f %%' % (acc_train * 100))

print('KNN Model Result with cv_scores_mean:: %.2f %%' % (knn_scores_mean * 100))


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
dt_pred_train = dt_model.predict(X_train)

# Perform cross validation on training set
dt_scores = cross_val_score(dt_model, X,y, cv=10)
dt_scores_mean = dt_scores.mean()
dt_scores_std = dt_scores.std()

# Compute the accuracy of the prediction
acc = float((y_test == dt_pred).sum()) / dt_pred.shape[0]
acc_train = float((y_train == dt_pred_train).sum()) / dt_pred_train.shape[0]

print('Test set accuracy in Decision Trees: %.2f %%' % (acc * 100))
print('Train set accuracy in Decision Trees: %.2f %%' % (acc_train * 100))
print('Decision Tree Model Result with cv_scores_mean:: %.2f %%' % (knn_scores_mean * 100))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train,y_train)
rfc_pred_train = rf_model.predict(X_train)
rfc_pred = rf_model.predict(X_test)

# Perform cross validation on training set
rf_scores = cross_val_score(rf_model, X,y, cv=10)
rf_scores_mean = rf_scores.mean()
rf_scores_std = rf_scores.std()

# Compute the accuracy of the prediction
acc_train = float((y_train == rfc_pred_train).sum()) / rfc_pred_train.shape[0]
acc = float((y_test == rfc_pred).sum()) / rfc_pred.shape[0]
print('Train set accuracy in Random Forest: %.2f %%' % (acc_train * 100))
print('Test set accuracy in Random Forest: %.2f %%' % (acc * 100))
print('Random Forest Model Result with cv_scores_mean:: %.2f %%' % (knn_scores_mean * 100))

# Visualisation of algorithm comparisons
plt.figure()
plt.plot(knn_pred,'gd', Label='KNN')
plt.plot(dt_pred,'+g', Label='Decision Trees')
plt.plot(rfc_pred,'^k', Label='Random Forest')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()

# Applying Forward feature selection
# Wrapper Feature Selection with Knn and dt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

sfs_knn = SFS(knn_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs_knn = sfs_knn.fit(X, y, custom_feature_names=feature_cols)

sfs_decision_tree = SFS(dt_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)
sfs_decision_tree = sfs_decision_tree.fit(X,y, custom_feature_names=feature_cols)

#print(sfs.subsets_)
print('KNN Model')
print('Selected features:', sfs_knn.k_feature_idx_)
print('Selected Features with names::', sfs_knn.k_feature_names_)
print('Selected Features with score::', sfs_knn.k_score_)
print(pd.DataFrame.from_dict(sfs_knn.get_metric_dict()).T)

print('Decision Tree Model')
print('Selected features:', sfs_decision_tree.k_feature_idx_)
print('Selected Features with names::', sfs_decision_tree.k_feature_names_)
print('Selected Features with score::', sfs_decision_tree.k_score_)
print(pd.DataFrame.from_dict(sfs_decision_tree.get_metric_dict()).T)

#plt.ylim([0.8,1])
fig_knn = plot_sfs(sfs_knn.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with KNN (w. StdDev)')
plt.grid()
plt.show()
#plt.ylim([0.8,1])
fig_dt = plot_sfs(sfs_decision_tree.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with descision tree (w. StdDev)')
plt.grid()
plt.show()

# Reduce independent columns
X_train_sfs_dt = sfs_decision_tree.transform(X)
X_train_sfs_knn = sfs_knn.transform(X)

# Perform Parameter Tuning with Stratified 10-fold Cross Validation and Grid Search
parameter_grid_dt = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}

parameter_grid_knn = {'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                   'weights': ['uniform', 'distance'],
                   'p': [1, 2, 3, 4, 5],
                   'metric': ['minkowski','manhattan']}

# Implementing parameter tuning using grid search and stratifiedKFold
print("Accuracy after applying parameter tuning")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

skf = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(dt_model, param_grid=parameter_grid_dt, cv=skf)
grid_search.fit(X_train_sfs_dt, y)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

skf = StratifiedKFold(n_splits=10, shuffle=True)
grid_search_knn = GridSearchCV(knn_model, param_grid=parameter_grid_knn, cv=skf, scoring='accuracy')
grid_search_knn.fit(X_train_sfs_knn, y)
print('Best score: {}'.format(grid_search_knn.best_score_))
print('Best parameters: {}'.format(grid_search_knn.best_params_))

# Use the best parameters above on same algos on test data
#KNN
knn_test = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', metric='minkowski', p=1, weights='uniform')
knn_test.fit(X_train_scaled,y_train)
knn_test_pred = knn_model.predict(X_test_scaled)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, knn_test_pred)
print('Confusion Matrix::',cm)
print(classification_report(y_test,knn_test_pred))

#Decision Tree
dt_test = DecisionTreeClassifier(criterion='gini', max_depth=4, splitter='random')
dt_test.fit(X_train,y_train)
dt_test_pred = dt_test.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, dt_test_pred)
print('Confusion Matrix::',cm)
print(classification_report(y_test,dt_test_pred))
