#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:55:54 2018

@author: shilinli
"""

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# kde plot of sepal_length versus sepal width for setosa
dt_iris = sns.load_dataset('iris')
sns.pairplot(dt_iris)
sns.kdeplot(data=dt_iris[dt_iris['species'] == 'setosa']['sepal_width'],data2=dt_iris[dt_iris['species'] == 'setosa']['sepal_length'],shade = True)

# Train Test Split
from sklearn.cross_validation import train_test_split
X = dt_iris.drop('species',axis=1)
Y = dt_iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Train the Support Vector Classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

# Evaluation and Prediction
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,pred)
classification_report(y_test,pred)

# Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_

pred_grid = grid.predict(X_test)
confusion_matrix(y_test,pred_grid)
classification_report(y_test,pred_grid)
