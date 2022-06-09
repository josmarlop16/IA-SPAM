from email_csv_parser import preproccessEmail
from naive_bayes_multinomial import NB_classifier_alpha_1, cv

def es_mensaje_no_deseado(path):
    # Preproccess the email
    transformed_text = preproccessEmail(path)
    # Vectorize the email
    vector_input = cv.transform(transformed_text)
    # Predict the email
    result = NB_classifier_alpha_1.predict(vector_input)
    # Print the email
    print(transformed_text)
    # [0:NotSpam, 1:Spam]
    if (result == 0):
        return False
    else:
        return True

print("¿Este correo es spam? ",es_mensaje_no_deseado(r"src/test/no_deseado/13"))
