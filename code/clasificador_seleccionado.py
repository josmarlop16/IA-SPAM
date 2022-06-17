from email_csv_parser import preproccessEmail
from vectorizer import bag_of_words_tokenizer
import pickle as cPickle


nbmA15 = cPickle.load(open("nbm.pickle", "rb"))
cv = cPickle.load(open("vectorizer.pickle", "rb"))

knn = cPickle.load(open("knn.pickle", "rb"))
feat = cPickle.load(open("knnvectorizer.pickle", "rb"))


def es_mensaje_no_deseado_NB(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    cv.set_params(analyzer=bag_of_words_tokenizer)
    vector_input = cv.transform(transformed_text)
    # Predict the email
    result = nbmA15.predict(vector_input)
    # [0:NotSpam, 1:Spam]
    if result == 0:
        return False
    else:
        return True


print(
    "Naive Bayes Detector -> ¿Este correo es spam? ",
    es_mensaje_no_deseado_NB(r"src/test/no_deseado/604"),
    "\n",
)


def es_mensaje_no_deseado_kNN(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    feat.set_params(analyzer=bag_of_words_tokenizer)
    vector_input = feat.transform(transformed_text)
    # Predict the email
    result = knn.predict(vector_input)
    # Print the email
    print(transformed_text)
    # [0:NotSpam, 1:Spam]
    if result == 0:
        return False
    else:
        return True


print(
    "kNN Detector -> ¿Este correo es spam? ",
    es_mensaje_no_deseado_kNN(r"src/test/no_deseado/604"),
    "\n",
)
