from base64 import decode
from email import policy
import nltk
from nltk.corpus import words
import pandas as pd
import email
import os
import csv


# Obtain the emails from the folder
listLegit = os.listdir(r"Enron-Spam\legítimo")
listSpam = os.listdir(r"Enron-Spam\no_deseado")


# Obtain english words
vocabulary = []

set_words = set(words.words())

# Email parser function
def parse_email(list, path):
    # Create a list to store the emails
    list_body = []
    # Iterate through the emails
    for i in list:
        # Open the email
        with open(path + "//" + i, encoding="latin-1") as f:
            msg = email.message_from_string(f.read())
            if msg.is_multipart():
                for part in msg.walk():

                    try:
                        payload = part.get_payload(
                            decode=True
                        )  # returns a bytes object
                        strtext = payload.decode("latin-1")
                    except:
                        strtext = part.get_payload(
                            decode=False
                        )  # returns a bytes object
                    list_body.append(strtext)

            else:
                try:
                    payload = msg.get_payload(decode=True)
                    strtext = payload.decode("latin-1")
                except:
                    strtext = msg.get_payload(decode=False)
                list_body.append(strtext)
    return list_body


# Function to get the vocabulary of the emails
def bag_of_words(list):
    for email in list:
        # iterate through each word in the email

        for word in email.split():
            if word.lower() not in vocabulary and word.lower() in set_words:
                vocabulary.append(word.lower())


# Parsing the emails
list_body_legit = parse_email(listLegit, r"Enron-Spam\legítimo")
list_body_spam = parse_email(listSpam, r"Enron-Spam\no_deseado")

# Get the vocabulary of the emails
# bag_of_words(list_body_legit)
# bag_of_words(list_body_spam)

############### Creating CSV files ###################
rowHeaders = ["Email", "isSpam"]

with open("vocabulary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow("Word")
    for i in range(len(vocabulary)):
        writer.writerow([vocabulary[i]])
    f.close()

with open("spam.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_spam)):
        writer.writerow([list_body_spam[i], "Yes"])
    f.close()

with open("legit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_legit)):
        writer.writerow([list_body_legit[i], "No"])
    f.close()
####################################################

dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
print(dataframe1.append(dataframe2))
