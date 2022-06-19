from vectorizer import bag_of_words_tokenizer
import pickle as cPickle
import email
import re
import time

## Los metodos y lineas de codigo comentadas son para la version de prueba de la practica ##

# count time of execution
start_time = time.time()

print("Loading pickles")
# nbmA15 = cPickle.load(open("nbm.pickle", "rb"))
# cv = cPickle.load(open("vectorizer.pickle", "rb"))

knn = cPickle.load(open("knn.pickle", "rb"))
feat = cPickle.load(open("knnvectorizer.pickle", "rb"))

print("Pickles loaded in", time.time() - start_time, "seconds")

# Email preproccessing function
def preproccessEmail(path):
    # Create a list to store the email
    list_body = []
    # Create a list to store the cleaned email
    cleaned_email = []
    # read the email
    with open(path, encoding="latin-1") as f:
        # get the body
        msg = email.message_from_file(f)
        if msg.is_multipart():
            for part in msg.walk():
                try:
                    payload = part.get_payload(decode=True)
                    strtext = payload.decode("latin-1")
                except:
                    strtext = part.get_payload(decode=False)
                list_body.append(strtext)
        else:
            try:
                payload = msg.get_payload(decode=True)
                strtext = payload.decode("latin-1")
            except:
                strtext = msg.get_payload(decode=False)
            list_body.append(strtext)
    for e in list_body:
        try:
            # To lower case
            e = e.lower()
            # Remove \n from an email
            e = e.replace("\n", "")
            # Remove html tags
            e = re.sub("<[^<>]+>", " ", e)
            # Normalize numbers
            e = re.sub("[0-9]+", "number", e)
            # Normalize URLs
            e = re.sub("(http|https)://[^\s]*", "httpAddress", e)
            # Normalize email addresses
            e = re.sub("[^\s]+@[^\s]+", "emailAddress", e)
            cleaned_email.append(e)
        except:
            # Normalize invalid emails (invalid type of email)
            if type(e) is list:
                e = ",".join(str(char) for char in e)
                cleaned_email.append(e)
    return cleaned_email

# def es_mensaje_no_deseado_NB(path):
#     # Preproccess the email
#     transformed_text = preproccessEmail(path)
#     # Vectorize the email
#     cv.set_params(analyzer=bag_of_words_tokenizer)
#     vector_input = cv.transform(transformed_text)
#     # Predict the email
#     result = nbmA15.predict(vector_input)
#     # [0:NotSpam, 1:Spam]
#     if result == 0:
#         return False
#     else:
#         return True

# print(
#     "Naive Bayes Detector -> ¿Este correo es spam? ",
#     es_mensaje_no_deseado_NB(r"src/test/no_deseado/604"),
#     "\n",
# )

def es_mensaje_no_deseado_kNN(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    feat.set_params(analyzer=bag_of_words_tokenizer)
    vector_input = feat.transform(transformed_text)
    # Predict the email
    result = knn.predict(vector_input)
    # [0:NotSpam, 1:Spam]
    if result == 0:
        return False
    else:
        return True

print(
    "kNN Detector -> ¿Este correo es spam? ",
    es_mensaje_no_deseado_kNN(r"src/test/legítimo/622"),
    "\n",
)
