import pandas as pd
import _pickle as cPickle
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
from tf_idf_vectorizer import train_tfidfvectorizer

print(" \n .·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·|| kNN Classifier ||·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.")

# Creating DataFrames
dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)

# remove null values
dataframe.dropna(inplace=True)

# Create a TF-IDF Vectorizer Object
features = train_tfidfvectorizer(dataframe["Email"])

# # convert to dataframe
features = pd.DataFrame(
    features.todense(), columns=tfidf_vectorizer.get_feature_names()
)
features.to_csv("features.csv", index=False)
# features = pd.read_csv("features.csv")
# Split the data into training and testing sets (60% training and 40% testing)
X_train, X_test, y_train, y_test = train_test_split(
    features, dataframe["isSpam"], test_size=0.4, random_state=42
)

# KNN Classifier
Knn_Classifier_n10 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
Knn_Classifier_n15 = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
Knn_Classifier_n20 = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)

# Let's see Classification report
print(
    "Classification report (n=10): \n",
    classification_report(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Classification report (n=15): \n",
    classification_report(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Classification report (n=20): \n",
    classification_report(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Confusion Matrix
print(
    "Confusion Matrix (n=10): \n ",
    confusion_matrix(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Confusion Matrix (n=15): \n ",
    confusion_matrix(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Confusion Matrix (n=20): \n ",
    confusion_matrix(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Accuracy score
print(
    "Accuracy score (n=10): \n",
    accuracy_score(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Accuracy score (n=15): \n",
    accuracy_score(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Accuracy score (n=20): \n",
    accuracy_score(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Precision score
print(
    "Precision score (n=10): \n",
    precision_score(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Precision score (n=15): \n",
    precision_score(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Precision score (n=20): \n",
    precision_score(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Recall score
print(
    "Recall score (n=10): \n", recall_score(y_test, Knn_Classifier_n10.predict(X_test))
)
print(
    "Recall score (n=15): \n", recall_score(y_test, Knn_Classifier_n15.predict(X_test))
)
print(
    "Recall score (n=20): \n", recall_score(y_test, Knn_Classifier_n20.predict(X_test))
)
# Let's see F1 score
print("F1 score (n=10): \n", f1_score(y_test, Knn_Classifier_n10.predict(X_test)))
print("F1 score (n=15): \n", f1_score(y_test, Knn_Classifier_n15.predict(X_test)))
print("F1 score (n=20): \n", f1_score(y_test, Knn_Classifier_n20.predict(X_test)))

print(
    ".·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·."
)
print(
    "¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨"
)
print("\n")

with open('knn.pickle', 'wb') as f:
   cPickle.dump(Knn_Classifier_n15, f, cPickle.HIGHEST_PROTOCOL)

