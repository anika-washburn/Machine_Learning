# Import libraries
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


dataset = pandas.read_csv("C:\\Users\\Anika\\Documents\\Case\\Sophomore Fall\\EECS297\\P2\\diabetes.csv")
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
knn.fit(X_train, Y_train)
predictionsKNN = knn.predict(X_validation)
print("KNeighbors:")
print(accuracy_score(Y_validation, predictionsKNN))
print(confusion_matrix(Y_validation, predictionsKNN))
print(classification_report(Y_validation, predictionsKNN))
#SVM
svm = SVC()
svm.fit(X_train, Y_train)
predictionsSVM = svm.predict(X_validation)
print("SVC:")
print(accuracy_score(Y_validation, predictionsSVM))
print(confusion_matrix(Y_validation, predictionsSVM))
print(classification_report(Y_validation, predictionsSVM))
