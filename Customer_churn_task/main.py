# -*- coding: utf-8 -*-
"""Customer_Churn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZxvbkjPQ7Ar2f26TyAb8zXHotPHJYbRX

# Importing the dataset and some libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("Churn_Modelling.csv")
x=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values


"""#Encoding Categorial data

## Geography row
"""


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[1])], remainder='passthrough')
newX=np.array(ct.fit_transform(x))
x=newX


"""##Gender row"""


le=LabelEncoder()
x[:,4]=le.fit_transform(x[:,4])

"""#Splitting the data"""



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

"""#Feature Scaling"""





sc1= StandardScaler()

x_train[:,5:9] = sc1.fit_transform(x_train[:,5:9])
x_test[:,5:9] = sc1.transform(x_test[:,5:9])

sc2=StandardScaler()

x_train_creditScore_reshaped = np.reshape(x_train[:,3], (-1, 1))
x_train_creditScore_scaled = sc2.fit_transform(x_train_creditScore_reshaped)
x_train[:,3] = np.ravel(x_train_creditScore_scaled)

x_test_creditScore_reshaped = np.reshape(x_test[:,3], (-1, 1))
x_test_creditScore_scaled = sc2.transform(x_test_creditScore_reshaped)
x_test[:,3] = np.ravel(x_test_creditScore_scaled)


sc3 = StandardScaler()

x_train_estimatedSalary_reshaped = np.reshape(x_train[:,-1], (-1, 1))
x_train_estimatedSalary_scaled = sc3.fit_transform(x_train_estimatedSalary_reshaped)
x_train[:,-1] = np.ravel(x_train_estimatedSalary_scaled)

x_test_estimatedSalary_reshaped = np.reshape(x_test[:,-1], (-1, 1))
x_test_estimatedSalary_scaled = sc3.transform(x_test_estimatedSalary_reshaped)
x_test[:,-1] = np.ravel(x_test_estimatedSalary_scaled)

def display_stats(model,model_name):
    print(model_name+"\n ")
    accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=5)

    print(" accuracy : {: .2f} %".format(accuracies.mean() * 100))
    print(" standard deviation : {: .2f} %".format(accuracies.std() * 100))

    scoring = make_scorer(f1_score)
    f1_scores = cross_val_score(estimator=model, X=x_train, y=y_train, cv=5, scoring=scoring)
    average_f1_score = f1_scores.mean()

    print("Average F1 score:", average_f1_score)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title(model_name)
    plt.show()

    print(accuracy_score(y_test, y_pred))


"""#Logistic Regression"""


LR = LogisticRegression()
LR.fit(x_train , y_train)
display_stats(LR,"logistic regression")


"""#KNN"""

print("K nearest neighbors \n ")
KNN = KNeighborsClassifier(n_neighbors=19 , p=2)
KNN.fit(x_train,y_train)

display_stats(KNN,"K nearest neighbors")


"""# SVM"""



SV_CL = SVC(kernel='poly')
SV_CL.fit(x_train , y_train)

display_stats(SV_CL,"Support vector machines")

"""#Naive Bayes"""


NB = GaussianNB()
NB.fit(x_train , y_train)

display_stats(NB,"naive bayes classifier")

"""#Decision Tree"""


DT = DecisionTreeClassifier(criterion="entropy")
DT.fit(x_train , y_train)

display_stats(DT,"Decision trees")

#from sklearn.tree import plot_tree
#fig = plt.figure(figsize=(25,20))
#_= plot_tree(DT , feature_names=data.columns,  class_names=["NO","YES"], filled=True , label= 'none', impurity= False)

"""#Random Forest"""


RF = RandomForestClassifier(n_estimators = 200 , criterion= 'entropy' )
RF.fit(x_train , y_train)
display_stats(RF,"Random forest")