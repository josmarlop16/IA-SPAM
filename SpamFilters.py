import pandas as pd
from nltk.corpus import stopwords
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

# nltk.download("stopwords")

############### Creating DataFrames ###################

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


def bag_of_words_tokenizer(email):
    # Remove punctuation
    no_punctuation = [char for char in email if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    # Remove stopwords
    clean_words = [
        word
        for word in no_punctuation.split()
        if word.lower() not in stopwords.words("english")
    ]
    return clean_words


print(dataframe["Email"].head().apply(bag_of_words_tokenizer))

# Create a bag of words vectorizer
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

# Prediction with the test set
print(NB_classifier.predict(X_test))

# Let's see the accuracy of the model
print(classification_report(y_test, NB_classifier.predict(X_test)))
# Confusion Matrix and accuracy
print(confusion_matrix(y_test, NB_classifier.predict(X_test)))
print(accuracy_score(y_test, NB_classifier.predict(X_test)))

# TO DO:
#
# Experimentar con otros valores de suavizado (parametro alpha)
# en MultinomialNB(alpha=x)

# Probar otras métricas de evaluacion

# Aplicar tecnicas para mejorar el modelo:
# - Preprocesamiento de datos: eliminacion de ruido,tokenizacion(ese creo q esta hecho), normalizacion
# - Uso de atributos derivados de los mensajes: indicacion de que el mensaje es una respuesta o envio
#   , tamaño del asunto y/o cuerpo del mensaje, presencia de etiquetas html en el cuerpo...
#
# Modelo tf-idf
