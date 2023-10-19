import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler
# loading dataset
data = pd.read_csv("spam.csv", header=None, encoding='latin-1', names=["label", "message","1","2","3"])
columns_to_drop = ["1","2","3"]
data.drop(columns=columns_to_drop,inplace=True)
# Convert label column
data["label"] = data["label"].map({"ham": 0, "spam": 1})
data.dropna(inplace=True)
# Split the data into training and testing sets

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data["message"])
ros = RandomOverSampler(random_state=42)
y = data.iloc[:,:-1].values
X_resampled,y_resampled = ros.fit_resample(X_tfidf, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled, test_size=0.2, random_state=42)
# Initialize the classifiers
naive_bayes = MultinomialNB()
logistic_regression = LogisticRegression()
support_vector_machine = SVC()

# Train the models
naive_bayes.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
support_vector_machine.fit(X_train, y_train)
# Evaluate the models
nb_predictions = naive_bayes.predict(X_test)
lr_predictions = logistic_regression.predict(X_test)
svm_predictions = support_vector_machine.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Support Vector Machine Accuracy:", svm_accuracy)

# Additional Evaluation Metrics
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

print("Support Vector Machine Classification Report:")
print(classification_report(y_test, svm_predictions))