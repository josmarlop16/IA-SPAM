import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import words
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Naive Bayes Multinomial Classifier

# nltk.download("stopwords")
# nltk.download("words")

# Creating DataFrames

dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

print(dataframe.shape)
print(dataframe.columns)

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)
print(dataframe.shape)

# remove null values
print(dataframe.isnull().sum())
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
        and word.lower() in words.words("english")
    ]
    return clean_words


# Example of the bag_of_words_tokenizer function applied to the first 5 emails in the dataframe
print(dataframe["Email"].head().apply(bag_of_words_tokenizer))

# Create a bag of words vectorizer and fit it to the dataframe
bag_of_words = CountVectorizer(analyzer=bag_of_words_tokenizer).fit_transform(
    dataframe["Email"]
)

# Split the data into training and testing sets (60% training and 40% testing)
# X stands for feature training dataset, y stands for target training dataset
X_train, X_test, y_train, y_test = train_test_split(
    bag_of_words, dataframe["isSpam"], test_size=0.4, random_state=42, shuffle=True
)

# Create a Polynomial Naive Bayes classifier and train it with the datasets
# Softened hyperparameter is alpha
NB_classifier = MultinomialNB().fit(X_train, y_train)
NB_classifier_alpha_2 = MultinomialNB(alpha=2).fit(X_train, y_train)
NB_classifier_alpha_3 = MultinomialNB(alpha=3).fit(X_train, y_train)

# Prediction with the test set
print("NB multinomial alpha=1: " + NB_classifier.predict(X_test))
print("NB multinomial alpha=2: " + NB_classifier_alpha_2.predict(X_test))
print("NB multinomial alpha=3: " + NB_classifier_alpha_3.predict(X_test))

# Let's see the accuracy of the model
print(
    "Classification report (alpha=1): "
    + classification_report(y_test, NB_classifier.predict(X_test))
)
print(
    "Classification report (alpha=2): "
    + classification_report(y_test, NB_classifier_alpha_2.predict(X_test))
)
print(
    "Classification report (alpha=3): "
    + classification_report(y_test, NB_classifier_alpha_3.predict(X_test))
)
# Let's see Confusion Matrix
print(
    "Confusion Matrix (alpha=1): /n "
    + confusion_matrix(y_test, NB_classifier.predict(X_test))
)
print(
    "Confusion Matrix (alpha=2): /n "
    + confusion_matrix(y_test, NB_classifier_alpha_2.predict(X_test))
)
print(
    "Confusion Matrix (alpha=3): /n "
    + confusion_matrix(y_test, NB_classifier_alpha_3.predict(X_test))
)
# Let's see Accuracy score
print(
    "Accuracy score (alpha=1): " + accuracy_score(y_test, NB_classifier.predict(X_test))
)
print(
    "Accuracy score (alpha=2): "
    + accuracy_score(y_test, NB_classifier_alpha_2.predict(X_test))
)
print(
    "Accuracy score (alpha=3): "
    + accuracy_score(y_test, NB_classifier_alpha_3.predict(X_test))
)

# Let's see Precision score
print(
    "Precision score (alpha=1): "
    + precision_score(y_test, NB_classifier.predict(X_test))
)
print(
    "Precision score (alpha=2): "
    + precision_score(y_test, NB_classifier_alpha_2.predict(X_test))
)
print(
    "Precision score (alpha=3): "
    + precision_score(y_test, NB_classifier_alpha_3.predict(X_test))
)

# Let's see Recall score
print("Recall score (alpha=1): " + recall_score(y_test, NB_classifier.predict(X_test)))
print(
    "Recall score (alpha=2): "
    + recall_score(y_test, NB_classifier_alpha_2.predict(X_test))
)
print(
    "Recall score (alpha=3): "
    + recall_score(y_test, NB_classifier_alpha_3.predict(X_test))
)

# Let's see F1 score
print("F1 score (alpha=1): " + f1_score(y_test, NB_classifier.predict(X_test)))
print("F1 score (alpha=2): " + f1_score(y_test, NB_classifier_alpha_2.predict(X_test)))
print("F1 score (alpha=3): " + f1_score(y_test, NB_classifier_alpha_3.predict(X_test)))

# TO DO:

# Aplicar tecnicas para mejorar el modelo:
# - Preprocesamiento de datos: eliminacion de ruido,tokenizacion(mejorar), normalizacion
# - Uso de atributos derivados de los mensajes: indicacion de que el mensaje es una respuesta o envio
#   , tamaño del asunto y/o cuerpo del mensaje, presencia de etiquetas html en el cuerpo...
