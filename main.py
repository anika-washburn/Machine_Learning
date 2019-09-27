# Import libraries
import copy
import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
import lightgbm as lgb



dataset = pandas.read_csv("diabetes.csv")
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(8, 9), sharex=False, sharey=False)
plt.show()

# histogram
dataset.hist()
plt.show()

# scatter plot
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
# All but last column
X = array[:, :8]
# Only last column
Y = array[:, 8]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'


#Spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# # Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions of validation dataset
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=6, criterion='entropy', random_state=seed)
params_rfc = {'n_estimators': [5, 10, 50, 100, 200]}
rf_gs = GridSearchCV(rfc, params_rfc, cv=5)
rf_gs.fit(X_train, Y_train)
rf_best = rf_gs.best_estimator_
predictionsRF = rf_best.predict(X_validation)
print("Random Forest:")
print(accuracy_score(Y_validation, predictionsRF))
print(confusion_matrix(Y_validation, predictionsRF))
print(classification_report(Y_validation, predictionsRF))
# LR
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictionsLR = lr.predict(X_validation)
print("Logistic Regression:")
print(accuracy_score(Y_validation, predictionsLR))
print(confusion_matrix(Y_validation, predictionsLR))
print(classification_report(Y_validation, predictionsLR))
# KNN
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(X_train, Y_train)
knn_best = knn_gs.best_estimator_
predictionsKNN = knn_best.predict(X_validation)
print("KNeighbors:")
print(accuracy_score(Y_validation, predictionsKNN))
print(confusion_matrix(Y_validation, predictionsKNN))
print(classification_report(Y_validation, predictionsKNN))
#SVM
svm = SVC()
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_svc = {'C': Cs, 'gamma': gammas}
svm_gs = GridSearchCV(svm, param_svc, cv=5)
svm_gs.fit(X_train, Y_train)
svm_best = svm_gs.best_estimator_
predictionsSVM = svm_best.predict(X_validation)
print("SVC:")
print(accuracy_score(Y_validation, predictionsSVM))
print(confusion_matrix(Y_validation, predictionsSVM))
print(classification_report(Y_validation, predictionsSVM))

# Ensemble of Classification Methods
estimators = [('rfc', rf_best), ('knn', knn_best), ('lr', lr), ('svm', svm_best)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(X_train, Y_train)
predictionsEnsemble = ensemble.predict(X_validation)
print("Ensemble:")
print(accuracy_score(Y_validation, predictionsEnsemble))
print(confusion_matrix(Y_validation, predictionsEnsemble))
print(classification_report(Y_validation, predictionsEnsemble))

# LightGBM
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_validation, Y_validation, reference=lgb_train)
params_lgb = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'l1'}, 'num_leaves': 31,
              'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
              'verbose': 0}
gbm = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
gbm.save_model('model.txt')
y_pred = gbm.predict(X_validation, num_iteration=gbm.best_iteration)
y_pred_rms = copy.copy(y_pred)
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
print('lgb: {}'.format(accuracy_score(Y_validation, y_pred)))
print('The rmse of prediction is: ', mean_squared_error(Y_validation, y_pred_rms)**0.5)

# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X, Y)
y_pred_NB = gnb.predict(X)
print('gnb: {}'.format(gnb.score(X, Y)))

