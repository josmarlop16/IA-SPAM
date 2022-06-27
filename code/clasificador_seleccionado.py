from vectorizer import bag_of_words_tokenizer
import pickle as cPickle
import email
import re

knn = cPickle.load(open("knn.pickle", "rb"))
feat = cPickle.load(open("knnvectorizer.pickle", "rb"))

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


def es_mensaje_no_deseado(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    feat.set_params(analyzer=bag_of_words_tokenizer)
    vector_input = feat.transform(transformed_text)
    # Predict the email
    result = knn.predict(vector_input)
    # [0(False)->NotSpam, 1(True)->Spam]
    if result == 0:
        return False
    else:
        return True


## Introducir la ruta del correo a evaluar en la llamada de la funcion
es_mensaje_no_deseado("Enron-Spam/no_deseado/34")
