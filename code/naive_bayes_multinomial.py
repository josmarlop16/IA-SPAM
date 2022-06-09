import string
import email_csv_parser
import pandas as pd
from nltk.corpus import stopwords, words
from sklearn import svm
from email_csv_parser import preproccessEmail
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# nltk.download("stopwords")
# nltk.download("words")


#.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·. Naive Bayes Multinomial Classifier .·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.
print(".·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·. Naive Bayes Multinomial Classifier .·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.")

# Creating DataFrames
dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)

# remove null values
dataframe.dropna(inplace=True)

# Function that obtain a bag of tokenized words
def bag_of_words_tokenizer(email):
    # Remove punctuation
    no_punctuation = [char for char in email if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    # Remove stopwords and check if word is in english dictionary
    clean_words = [
        word
        for word in no_punctuation.split()
        if word.lower() not in stopwords.words("english")
    ]
    return clean_words

# Example of the bag_of_words_tokenizer function applied to the first 5 emails in the dataframe
words = dataframe["Email"].apply(bag_of_words_tokenizer);
words.to_csv("words.csv", index=False)

# Create a bag of words vectorizer and fit it to the dataframe
cv = CountVectorizer(analyzer=bag_of_words_tokenizer)
bag_of_words = cv.fit_transform(dataframe["Email"])

# Split the data into training and testing sets (60% training and 40% testing)
# X stands for feature training dataset, y stands for target training dataset
X_train, X_test, y_train, y_test = train_test_split( bag_of_words, dataframe["isSpam"], test_size=0.4, random_state=42, shuffle=True)

# Create a Polynomial Naive Bayes classifier and train it with the datasets
# Softened hyperparameter is alpha
NB_classifier_alpha_1 = MultinomialNB().fit(X_train, y_train)
NB_classifier_alpha_2 = MultinomialNB(alpha=2).fit(X_train, y_train)
NB_classifier_alpha_3 = MultinomialNB(alpha=3).fit(X_train, y_train)

# Let's see the accuracy of the model
print("Classification report (alpha=1): \n", classification_report(y_test, NB_classifier_alpha_1.predict(X_test)))
print("Classification report (alpha=2): \n", classification_report(y_test, NB_classifier_alpha_2.predict(X_test)))
print("Classification report (alpha=3): \n", classification_report(y_test, NB_classifier_alpha_3.predict(X_test)))
# Let's see Confusion Matrix
print("Confusion Matrix (alpha=1): \n", confusion_matrix(y_test, NB_classifier_alpha_1.predict(X_test)))
print("Confusion Matrix (alpha=2): \n", confusion_matrix(y_test, NB_classifier_alpha_2.predict(X_test)))
print("Confusion Matrix (alpha=3): \n", confusion_matrix(y_test, NB_classifier_alpha_3.predict(X_test)))
# Let's see Accuracy score
print("Accuracy score (alpha=1): \n", accuracy_score(y_test, NB_classifier_alpha_1.predict(X_test)))
print("Accuracy score (alpha=2): \n", accuracy_score(y_test, NB_classifier_alpha_2.predict(X_test)))
print("Accuracy score (alpha=3): \n", accuracy_score(y_test, NB_classifier_alpha_3.predict(X_test)))
# Let's see Precision score
print("Precision score (alpha=1): \n", precision_score(y_test, NB_classifier_alpha_1.predict(X_test)))
print("Precision score (alpha=2): \n", precision_score(y_test, NB_classifier_alpha_2.predict(X_test)))
print("Precision score (alpha=3): \n", precision_score(y_test, NB_classifier_alpha_3.predict(X_test)))
# Let's see Recall score
print("Recall score (alpha=1): \n", recall_score(y_test, NB_classifier_alpha_1.predict(X_test)))
print("Recall score (alpha=2): \n", recall_score(y_test, NB_classifier_alpha_2.predict(X_test)))
print("Recall score (alpha=3): \n", recall_score(y_test, NB_classifier_alpha_3.predict(X_test)))
# Let's see F1 score
print("F1 score (alpha=1): \n", f1_score(y_test, NB_classifier_alpha_1.predict(X_test)))
print("F1 score (alpha=2): \n", f1_score(y_test, NB_classifier_alpha_2.predict(X_test)))
print("F1 score (alpha=3): \n", f1_score(y_test, NB_classifier_alpha_3.predict(X_test)))

# TO DO:
# Aplicar tecnicas para mejorar el modelo:
# - Preprocesamiento de datos: eliminacion de ruido,tokenizacion(mejorar), normalizacion
# - Uso de atributos derivados de los mensajes: indicacion de que el mensaje es una respuesta o envio
#   , tamaño del asunto y/o cuerpo del mensaje, presencia de etiquetas html en el cuerpo...
