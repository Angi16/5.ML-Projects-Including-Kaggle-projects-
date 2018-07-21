
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

#Load data

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Summarize Dataset

#Dimensions of dataset

# shape
print(dataset.shape)

# output:
#(150, 5)

#Peek at the Data
	
# head
print(dataset.head(20))

#Statistical Summary

# descriptions
print(dataset.describe())

#Class Distribution

print(dataset.groupby('class').size())

#We can see that each class has the same number of instances (50 or 33% of the dataset).

'''	
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50

'''

#Data Visualization

# Univariate Plots
#Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
	
# histograms
dataset.hist()
plt.show()
	
# scatter plot matrix
scatter_matrix(dataset)
plt.show()


#Evaluate Some Algorithms

'''

    1.Separate out a validation dataset.
    2.Set-up the test harness to use 10-fold cross validation.
    3.Build 5 different models to predict species from flower measurements
    4.Select the best model.

'''

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

#Build Models

'''

    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM).
'''

#build and evaluate our five models:

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []2
3
4
5
6
7
	
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

2
3
4
5
6
7
	
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
#Select Best Model

#Running the example above, we get the following raw results:

'''
Final Result:-

.9
 
[[ 7  0  0]
 [ 0 11  1]
 [ 0  2  9]]
 
             precision    recall  f1-score   support
 
Iris-setosa       1.00      1.00      1.00         7
Iris-versicolor   0.85      0.92      0.88        12
Iris-virginica    0.90      0.82      0.86        11
 
avg / total       0.90      0.90      0.90        30

'''