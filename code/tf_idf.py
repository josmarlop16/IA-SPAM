import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import words
import string
from sklearn.cluster import k_means
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Creating DataFrames
dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)
print(dataframe.shape)

# remove null values
print(dataframe.isnull().sum())
dataframe.dropna(inplace=True)

# Create a TF-IDF Vectorizer Object
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
# Fit the TF-IDF vectorizer to the training data
features = tfidf_vectorizer.fit_transform(dataframe["Email"])

# Split the data into training and testing sets (60% training and 40% testing)
X_train, X_test, y_train, y_test = train_test_split(
    features, dataframe["isSpam"], test_size=0.4, random_state=42
)

# KNN Classifier
Knn_Classifier_n3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
Knn_Classifier_n4 = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
Knn_Classifier_n5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

print(
    "Classification report (n=3): \n",
    classification_report(y_test, Knn_Classifier_n3.predict(X_test)),
)
print(
    "Classification report (n=4): \n",
    classification_report(y_test, Knn_Classifier_n4.predict(X_test)),
)
print(
    "Classification report (n=5): \n",
    classification_report(y_test, Knn_Classifier_n5.predict(X_test)),
)
# Let's see Confusion Matrix
print(
    "Confusion Matrix (n=3): \n ",
    confusion_matrix(y_test, Knn_Classifier_n3.predict(X_test)),
)
print(
    "Confusion Matrix (n=4): \n ",
    confusion_matrix(y_test, Knn_Classifier_n4.predict(X_test)),
)
print(
    "Confusion Matrix (n=5): \n ",
    confusion_matrix(y_test, Knn_Classifier_n5.predict(X_test)),
)
# Let's see Accuracy score
print(
    "Accuracy score (n=3): \n",
    accuracy_score(y_test, Knn_Classifier_n3.predict(X_test)),
)
print(
    "Accuracy score (n=4): \n",
    accuracy_score(y_test, Knn_Classifier_n4.predict(X_test)),
)
print(
    "Accuracy score (n=5): \n",
    accuracy_score(y_test, Knn_Classifier_n5.predict(X_test)),
)
# Let's see Precision score
print(
    "Precision score (n=3): \n",
    precision_score(y_test, Knn_Classifier_n3.predict(X_test)),
)
print(
    "Precision score (n=4): \n",
    precision_score(y_test, Knn_Classifier_n4.predict(X_test)),
)
print(
    "Precision score (n=5): \n",
    precision_score(y_test, Knn_Classifier_n5.predict(X_test)),
)
# Let's see Recall score
print("Recall score (n=3): \n", recall_score(y_test, Knn_Classifier_n3.predict(X_test)))
print("Recall score (n=4): \n", recall_score(y_test, Knn_Classifier_n4.predict(X_test)))
print("Recall score (n=5): \n", recall_score(y_test, Knn_Classifier_n5.predict(X_test)))
# Let's see F1 score
print("F1 score (n=3): \n", f1_score(y_test, Knn_Classifier_n3.predict(X_test)))
print("F1 score (n=4): \n", f1_score(y_test, Knn_Classifier_n4.predict(X_test)))
print("F1 score (n=5): \n", f1_score(y_test, Knn_Classifier_n5.predict(X_test)))

