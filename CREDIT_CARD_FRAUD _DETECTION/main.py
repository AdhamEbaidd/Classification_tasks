import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("fraudTrain.csv")
y_train = data.iloc[:, -1].values


# Dropping irrelevant features
columns_to_drop = ["Unnamed: 0","cc_num","first","last","street","merchant","trans_num","unix_time","zip"]
data.drop(columns=columns_to_drop,inplace=True)



# dealing with date features
data['dob'] = pd.to_datetime(data['dob'])
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_date_trans_time'] = data['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

# Get transaction month and year only
data["trans_month"] = data["trans_date_trans_time"].dt.month
data["trans_year"] = data["trans_date_trans_time"].dt.year

# turning dob feature into age
data["age"] = (data["trans_date_trans_time"]- data["dob"]).apply(lambda x: int(x.days / 365))

# getting distance bewteen seller and home
data['latitudinal_distance'] = abs(round(data['merch_lat'] - data['lat'],3))
data['longitudinal_distance'] = abs(round(data['merch_long'] - data['long'],3))

# dropping un-needed columns
drop_columns = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','state','is_fraud']
data.drop(columns=drop_columns,inplace=True)

# categorical feature encoding
data = pd.get_dummies(data , columns=['category'] , prefix='category')
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])

# loading values into feature dataframe
x_train = data.iloc[:, :].values

#doing the same to the testing set
testing_data=pd.read_csv("fraudTest.csv")
y_test = testing_data.iloc[ : , -1].values
testing_data.drop(columns=columns_to_drop,inplace=True)
testing_data['dob'] = pd.to_datetime(testing_data['dob'])
testing_data['trans_date_trans_time'] = pd.to_datetime(testing_data['trans_date_trans_time'])
testing_data['trans_date_trans_time'] = testing_data['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
testing_data['trans_date_trans_time'] = pd.to_datetime(testing_data['trans_date_trans_time'])

# Get transaction month and year only
testing_data["trans_month"] = testing_data["trans_date_trans_time"].dt.month
testing_data["trans_year"] = testing_data["trans_date_trans_time"].dt.year

# turning dob feature into age
testing_data["age"] = (testing_data["trans_date_trans_time"]- testing_data["dob"]).apply(lambda x: int(x.days / 365))

# getting distance bewteen seller and home
testing_data['latitudinal_distance'] = abs(round(testing_data['merch_lat'] - testing_data['lat'],3))
testing_data['longitudinal_distance'] = abs(round(testing_data['merch_long'] - testing_data['long'],3))

# dropping un-needed columns
testing_data.drop(columns=drop_columns,inplace=True)

# categorical feature encoding
testing_data = pd.get_dummies(testing_data , columns=['category'] , prefix='category')
le_2 = LabelEncoder()
testing_data["gender"] = le_2.fit_transform(testing_data["gender"])

# loading values into feature dataframe
x_test = testing_data.iloc[:, :].values

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

KNN = KNeighborsClassifier(n_neighbors=5 , p=2)
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



"""#Random Forest"""


RF = RandomForestClassifier(n_estimators = 100 , criterion= 'entropy' )
RF.fit(x_train , y_train)
display_stats(RF,"Random forest")