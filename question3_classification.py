#Q7: Does the systematic banking crisis , micro and macro economic indicators and number of protests results in change in political leadership?

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
print(orig_dataset.columns)

# Handling missing data
orig_dataset = orig_dataset.fillna(0)
print(orig_dataset.columns)
print(orig_dataset.get_dtype_counts())

Europe = ['AUT','BEL','CHE','DEU','DNK','ESP','FIN','FRA','GBR','GRC','IRL','ITA','NLD','NOR','PRT','SWE']
european_nations = orig_dataset[orig_dataset.iso.isin(Europe)]

#original dataset european_nations.fillna(0)
feature_cols=['rgdp', 'crisisJST', 'pk_fin', 'pk_norm', 'pk_dis', 'cpi', 'vetopl','protests']
X = orig_dataset[feature_cols]
y = orig_dataset.turnover

#model selection using Train/Test splits
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled,y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_pred_train = knn_model.predict(X_train_scaled)

# Perform cross validation on dataset
knn_scores = cross_val_score(knn_model, X,y, cv=10)
knn_scores_mean = knn_scores.mean()
knn_scores_std = knn_scores.std()

# Compute the accuracy of the prediction
acc = float((y_test == knn_pred).sum()) / knn_pred.shape[0]
acc_train = float((y_train == knn_pred_train).sum()) / knn_pred_train.shape[0]
print('Train set accuracy in KNN: %.2f %%' % (acc_train * 100))
print('Test set accuracy in KNN: %.2f %%' % (acc * 100))
print('KNN Model Result with cv_scores_mean:: %.2f %%' % (knn_scores_mean * 100))

# #Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
dt_pred_train = dt_model.predict(X_train)

# Perform cross validation on dataset
dt_scores = cross_val_score(dt_model, X,y, cv=10)
dt_scores_mean = dt_scores.mean()
dt_scores_std = dt_scores.std()

# Compute the accuracy of the prediction
acc = float((y_test == dt_pred).sum()) / dt_pred.shape[0]
acc_train = float((y_train == dt_pred_train).sum()) / dt_pred_train.shape[0]
print('Train set accuracy in Decision Trees: %.2f %%' % (acc_train * 100))
print('Test set accuracy in Decision Trees: %.2f %%' % (acc * 100))
print('Decision Tree Model Result with cv_scores_mean:: %.2f %%' % (dt_scores_mean * 100))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train,y_train)
rfc_pred = rf_model.predict(X_test)
rfc_pred_train = rf_model.predict(X_train)

# Perform cross validation on dataset
rf_scores = cross_val_score(rf_model, X,y, cv=10)
rf_scores_mean = rf_scores.mean()
rf_scores_std = rf_scores.std()

# Compute the accuracy of the prediction
acc = float((y_test == rfc_pred).sum()) / rfc_pred.shape[0]
acc_train = float((y_train == rfc_pred_train).sum()) / rfc_pred_train.shape[0]

print('Train set accuracy in Random Forest: %.2f %%' % (acc_train * 100))
print('Test set accuracy in Random Forest: %.2f %%' % (acc * 100))
print('Random Forest Model Result with cv_scores_mean:: %.2f %%' % (rf_scores_mean * 100))

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

sfs_rf = SFS(rf_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)
sfs_rf = sfs_rf.fit(X,y, custom_feature_names=feature_cols)

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

print('Random Forest Model')
print('Selected features:', sfs_rf.k_feature_idx_)
print('Selected Features with names::', sfs_rf.k_feature_names_)
print('Selected Features with score::', sfs_rf.k_score_)
print(pd.DataFrame.from_dict(sfs_rf.get_metric_dict()).T)

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
#plt.ylim([0.8,1])
fig_rf = plot_sfs(sfs_rf.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with random forest (w. StdDev)')
plt.grid()
plt.show()

#Reduce independent columns
X_train_sfs_dt = sfs_decision_tree.transform(X)
X_train_sfs_knn = sfs_knn.transform(X)
X_train_sfs_rf = sfs_rf.transform(X)

# Perform Parameter Tuning with Stratified 10-fold Cross Validation and Grid Search
parameter_grid_dt = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}

parameter_grid_knn = {'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                   'weights': ['uniform', 'distance'],
                   'p': [1, 2, 3, 4, 5],
                   'metric': ['minkowski','manhattan']}

parameter_grid_rf = {'criterion': ['gini', 'entropy'],
                    'max_features' : ['auto', 'sqrt'],
                   'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}

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

skf = StratifiedKFold(n_splits=10, shuffle=True)
grid_search_rf = GridSearchCV(rf_model, param_grid=parameter_grid_rf, cv=skf, scoring='accuracy')
grid_search_rf.fit(X_train_sfs_rf, y)
print('Best score: {}'.format(grid_search_rf.best_score_))
print('Best parameters: {}'.format(grid_search_rf.best_params_))

# Use the best parameters above on same algos on test data
#KNN
print('KNN Model Prediction Accuracy')
knn_test = KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='minkowski', p=2, weights='uniform')
knn_test.fit(X_train_scaled,y_train)
knn_test_pred = knn_test.predict(X_test_scaled)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# cm = confusion_matrix(y_test, knn_test_pred)
# print('Confusion Matrix::',cm)
# print(classification_report(y_test,knn_test_pred))
auc = roc_auc_score(y_test,knn_test_pred)
print(auc)
knn_fpr,knn_tpr,threshold = roc_curve(y_test,knn_test.predict_proba(X_test_scaled)[:,1])

#Decision Tree
print('Decision Tree Model Prediction Accuracy')
dt_test = DecisionTreeClassifier(criterion='entropy', max_depth=8, splitter='best')
dt_test.fit(X_train,y_train)
dt_test_pred = dt_test.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# cm = confusion_matrix(y_test, dt_test_pred)
# print('Confusion Matrix::',cm)
# print(classification_report(y_test,dt_test_pred))
auc = roc_auc_score(y_test,dt_test_pred)
print(auc)
fpr,tpr,threshold = roc_curve(y_test,dt_test.predict_proba(X_test)[:,1])
#print('threshold::',threshold)

#Random Forest
print('Random Forest Model Prediction Accuracy')
rf_test = RandomForestClassifier( criterion= 'gini', max_depth = 1, max_features ='auto', n_estimators = 100)
rf_test.fit(X_train,y_train)
rf_test_pred = rf_test.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# cm = confusion_matrix(y_test, rf_test_pred)
# print('Confusion Matrix::',cm)
# print(classification_report(y_test,rf_test_pred))
auc = roc_auc_score(y_test,rf_test_pred)
print(auc)
rf_fpr,rf_tpr,threshold = roc_curve(y_test,rf_test.predict_proba(X_test)[:,1])

plt.plot(knn_fpr,knn_tpr, Label='KNN')
plt.plot(fpr,tpr, label='Decision Tree')
plt.plot(rf_fpr,rf_tpr, Label='Random Forest')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid(True)
plt.legend(loc="best")
plt.show()