import email_csv_parser
from naive_bayes_multinomial import NB_classifier_alpha_1, bag_of_words_tokenizer
from sklearn.feature_extraction.text import CountVectorizer

def es_mensaje_no_deseado(path):
    parsedEmail = email_csv_parser.parse_email2(path)
    cleanedEmail = email_csv_parser.clean_email(parsedEmail)
    cv = CountVectorizer(analyzer=bag_of_words_tokenizer)
    pred = cv.transform(cleanedEmail)
    prediction = NB_classifier_alpha_1.predict(pred)
    if prediction[0] == 0:
        return False
    else:
        return True

print(es_mensaje_no_deseado(r"Enron-Spam/legítimo/0"))
