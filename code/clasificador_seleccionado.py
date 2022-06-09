from email_csv_parser import preproccessEmail
from naive_bayes_multinomial import NB_classifier_alpha_1, NB_classifier_alpha_2, NB_classifier_alpha_3, cv
from tf_idf import tfidf_vectorizer, Knn_Classifier_n5

def es_mensaje_no_deseado(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    vector_input = cv.transform(transformed_text)
    # Predict the email
    result = NB_classifier_alpha_3.predict(vector_input)
    # Print the email
    print(transformed_text)
    # [0:NotSpam, 1:Spam]
    if (result == 0):
        return False
    else:
        return True

print("Naive Bayes Detector -> ¿Este correo es spam? ",es_mensaje_no_deseado(r"src/test/no_deseado/9"), "\n")

def es_mensaje_no_deseado_tf_idf(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    vector_input = tfidf_vectorizer.transform(transformed_text)
    # Predict the email
    result = Knn_Classifier_n5.predict(vector_input)
    # Print the email
    print(transformed_text)
    # [0:NotSpam, 1:Spam]
    if (result == 0):
        return False
    else:
        return True

print("tf-idf Detector -> ¿Este correo es spam? ",es_mensaje_no_deseado_tf_idf(r"src/test/no_deseado/9"), "\n")
