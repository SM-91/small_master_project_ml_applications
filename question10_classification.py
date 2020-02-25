#Q10: how does elections effect political stability among western nations?

#Importing Libs
import pandas as pd
import numpy as np
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
orig_dataset = orig_dataset.interpolate(method='linear',limit_area ='inside')
orig_dataset = orig_dataset.fillna(0)
print(orig_dataset.get_dtype_counts())

Europe = ['AUT','BEL','CHE','DEU','DNK','ESP','FIN','FRA','GBR','GRC','IRL','ITA','NLD','NOR','PRT','SWE']
North_America = ['USA','CAN']

european_region = orig_dataset[orig_dataset.iso.isin(Europe)]
american_region = orig_dataset[orig_dataset.iso.isin(North_America)]

feature_cols=['govvote', 'oppvote', 'frac', 'partycount', 'right', 'left', 'extr', 'protests', 'protestsdev', 'demosdev', 'riotsdev', 'strikesdev']
X = american_region[feature_cols]
y = american_region.election

#model selection using Train/Test splits
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Learning Models
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
logic_pred_train=logistic_model.predict(X_train)
logic_pred=logistic_model.predict(X_test)

# Perform cross validation on training set
logic_scores = cross_val_score(logistic_model, X,y, cv=10)
logic_scores_mean = logic_scores.mean()
logic_scores_std = logic_scores.std()
# Compute the accuracy of the prediction
acc_train = float((y_train == logic_pred_train).sum()) / logic_pred_train.shape[0]
acc = float((y_test == logic_pred).sum()) / logic_pred.shape[0]

print('Train set accuracy in Logistic Regression: %.2f %%' % (acc_train * 100))
print('Test set accuracy in Logistic Regression: %.2f %%' % (acc * 100))
print('Logistic Model Result with cv_scores_mean:: %.2f %%' % (logic_scores_mean * 100))


#SVM
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_pred_train = svm_model.predict(X_train_scaled)
svm_pred = svm_model.predict(X_test_scaled)

# Perform cross validation on training set
svm_scores = cross_val_score(svm_model, X,y, cv=10)
svm_scores_mean = svm_scores.mean()
svm_scores_std = svm_scores.std()

# Compute the accuracy of the prediction
acc_train = float((y_train == svm_pred_train).sum()) / svm_pred_train.shape[0]
acc = float((y_test == svm_pred).sum()) / svm_pred.shape[0]
print('Train set accuracy in SVM: %.2f %%' % (acc_train * 100))
print('Test set accuracy in SVM: %.2f %%' % (acc * 100))
print('SVM Model Result with cv_scores_mean:: %.2f %%' % (svm_scores_mean * 100))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB

nb_model = GaussianNB() #BernoulliNB
nb_model.fit(X_train,y_train)
nb_model_pred_train = nb_model.predict(X_train)
nb_model_pred = nb_model.predict(X_test)

# Perform cross validation on training set
nb_scores = cross_val_score(nb_model, X,y, cv=10)
nb_scores_mean = nb_scores.mean()
nb_scores_std = nb_scores.std()

# Compute the accuracy of the prediction
acc_train = float((y_train == nb_model_pred_train).sum()) / nb_model_pred_train.shape[0]
acc = float((y_test == nb_model_pred).sum()) / nb_model_pred.shape[0]

print('Train set accuracy in Naive Bayes: %.2f %%' % (acc_train * 100))
print('Test set accuracy in Naive Bayes: %.2f %%' % (acc * 100))
print('Naive Bayes Model Result with cv_scores_mean:: %.2f %%' % (nb_scores_mean * 100))


# Visualisation of algorithm comparisons
plt.figure()
plt.plot(logic_pred, 'ys', label='Logistic Regression')
plt.plot(svm_pred, 'b^', label='Support Vector Machine')
plt.plot(nb_model_pred,'*m', Label='Naive Bayes')
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

sfs_log = SFS(logistic_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs_log = sfs_log.fit(X, y, custom_feature_names=feature_cols)

sfs_svm = SFS(svm_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)
sfs_svm = sfs_svm.fit(X,y, custom_feature_names=feature_cols)

sfs_nb = SFS(nb_model,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs_nb = sfs_nb.fit(X, y, custom_feature_names=feature_cols)

# Reduce independent columns
X_train_sfs_log = sfs_log.transform(X)
X_train_sfs_svm = sfs_svm.transform(X)
X_train_sfs_nb = sfs_nb.transform(X)


#print(sfs.subsets_)
print('Logistic Regression Model')
print('Selected features:', sfs_log.k_feature_idx_)
print('Selected Features with names::', sfs_log.k_feature_names_)
print('Selected Features with score::', sfs_log.k_score_)
print(pd.DataFrame.from_dict(sfs_log.get_metric_dict()).T)

print('SVM Model')
print('Selected features:', sfs_svm.k_feature_idx_)
print('Selected Features with names::', sfs_svm.k_feature_names_)
print('Selected Features with score::', sfs_svm.k_score_)
print(pd.DataFrame.from_dict(sfs_svm.get_metric_dict()).T)

print('Naive Bayes Model')
print('Selected features:', sfs_nb.k_feature_idx_)
print('Selected Features with names::', sfs_nb.k_feature_names_)
print('Selected Features with score::', sfs_nb.k_score_)
print(pd.DataFrame.from_dict(sfs_nb.get_metric_dict()).T)

fig_log = plot_sfs(sfs_log.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with Logistic Regression (w. StdDev)')
plt.grid()
plt.show()

fig_svm = plot_sfs(sfs_svm.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with SVM (w. StdDev)')
plt.grid()
plt.show()

fig_nb = plot_sfs(sfs_nb.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection with Naive Bayes (w. StdDev)')
plt.grid()
plt.show()

# Implementing parameter tuning using grid search and stratifiedKFold
parameter_grid_lr = {'C': np.logspace(-3,3,7),
                   'penalty': ['l1', 'l2']} # l1 lasso l2 ridge

print("Accuracy after applying parameter tuning")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

skf = StratifiedKFold(n_splits=10, shuffle=True)
grid_search_knn = GridSearchCV(logistic_model, param_grid=parameter_grid_lr, cv=skf, scoring='accuracy')
grid_search_knn.fit(X_train_sfs_log, y)
print('Best score: {}'.format(grid_search_knn.best_score_))
print('Best parameters: {}'.format(grid_search_knn.best_params_))

#LR
print('KNN Model Prediction Accuracy')
logistic_model_test = LogisticRegression( C = 0.001, penalty = 'l1')
logistic_model_test.fit(X_train_scaled,y_train)
logistic_model_test_pred = logistic_model_test.predict(X_test_scaled)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, logistic_model_test_pred)
print('Confusion Matrix::',cm)
print(classification_report(y_test,logistic_model_test_pred))
