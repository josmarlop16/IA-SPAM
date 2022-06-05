from email import policy
import nltk
from nltk.corpus import words
import pandas as pd
import email
import os
import csv

nltk.download("punkt")
# Obtain the emails from the folder
listLegit = os.listdir(r"Enron-Spam\legítimo")
listSpam = os.listdir(r"Enron-Spam\no_deseado")


# Obtain english words
vocabulary = []
nltk.download("words")
set_words = set(words.words())

# Email parser function
def parse_email(list, path):
    list_body = []
    for i in list:
        with open(path + "\\" + i, "rb") as fp:
            msg = email.message_from_binary_file(fp, policy=policy.default)
            try:
                body = msg.get_body(
                    preferencelist=("related", "html", "plain")
                ).get_payload()
            except:
                print("Error")
                body = msg.get_payload()
                # .decode("utf-8")
                # list_body.append(body)
            list_body.append(body)
    return list_body


# Function to get the vocabulary of the emails
def bag_of_words(list):
    for email in list:
        # iterate through each word in the email

        for word in email.split():
            if word.lower() not in vocabulary and word.lower() in set_words:
                vocabulary.append(word.lower())


# Parsing the emails
# list_body_legit = parse_email(listLegit, r"Enron-Spam\legítimo")
list_body_spam = parse_email(listSpam, r"Enron-Spam\no_deseado")

# Get the vocabulary of the emails
# bag_of_words(list_body_legit)
bag_of_words(list_body_spam)

############### Creating CSV files ###################
rowHeaders = ["Email", "isSpam"]

with open("vocabulary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow("Word")
    for i in range(len(vocabulary)):
        writer.writerow([vocabulary[i]])
    f.close()

with open("spam.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_spam)):
        writer.writerow([list_body_spam[i]])
    f.close()

# with open("legit.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(rowHeaders)
#     for i in range(len(list_body_legit)):
#         writer.writerow([list_body_legit[i], "No"])
#     f.close()
#####################################################
