import email
import os
import csv
from bs4 import BeautifulSoup
import re

# splitfolders.fixed(
#     r"Enron-Spam/",
#     output="src",
#     seed=42,
#     fixed=(500, 500),
#     oversample=False,
#     group_prefix=None,
# )

# Obtain the emails from the folder
listLegit = os.listdir(r"Enron-spam/legítimo")
listSpam = os.listdir(r"Enron-spam/no_deseado")

# Email parser function
def parse_email(list, path):
    # Create a list to store the emails
    list_body = []
    list_hasHTML = []

    # Iterate through the emails
    for i in list:
        # read the email
        with open(path + "/" + i, encoding="latin-1") as f:
            # get the body
            msg = email.message_from_string(f.read())
            if msg.is_multipart():
                for part in msg.walk():
                    try:
                        payload = part.get_payload(decode=True)
                        # returns a bytes object
                        strtext = payload.decode("latin-1")
                        hasHTML = bool(
                            BeautifulSoup(msg.get_payload(), "html.parser").find()
                        )
                        list_hasHTML.append(hasHTML)
                    except:
                        strtext = part.get_payload(decode=False)
                        list_hasHTML.append(False)
                        # returns a bytes object
                    list_body.append(strtext)
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    strtext = payload.decode("latin-1")
                    hasHTML = bool(
                        BeautifulSoup(msg.get_payload(), "html.parser").find()
                    )
                    list_hasHTML.append(hasHTML)
                except:
                    strtext = msg.get_payload(decode=False)
                    hasHTML = bool(
                        BeautifulSoup(msg.get_payload(), "html.parser").find()
                    )
                    list_hasHTML.append(hasHTML)
                list_body.append(strtext)
    return list_body, list_hasHTML


# Parsing the emails
list_body_legit = parse_email(listLegit, r"Enron-spam/legítimo")
list_body_spam = parse_email(listSpam, r"Enron-spam/no_deseado")

# Email cleaner function
def clean_email(emailList):
    # remove html tags and \n from an email, addresses, numbers and links
    # list to store the cleaned email
    cleaned_email = []
    for email in emailList:
        try:
            # To lower case
            email = email.lower()
            # Remove html tags
            email = re.sub("<[^<>]+>", " ", email)
            # email = striphtml(email)
            # email = email.replace("<", "").replace(">", "").replace("/n", "")
            # Normalize numbers
            email = re.sub("[0-9]+", "number", email)
            # Normalize URLs
            email = re.sub("(http|https)://[^\s]*", "httpAddress", email)
            # Normalize email addresses
            email = re.sub("[^\s]+@[^\s]+", "emailAddress", email)

            # se pueden añadir mas cosas

            cleaned_email.append(email)
        except:
            # Normalize invalid emails
            if type(email) is list:
                email = ",".join(str(char) for char in email)
                cleaned_email.append(email)

    return cleaned_email


# Cleaning the emails
list_body_legit_cleaned = clean_email(list_body_legit)
list_body_spam_cleaned = clean_email(list_body_spam)

############### Creating CSV files ###################
rowHeaders = ["Email", "isSpam"]

with open("spam.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_spam_cleaned)):
        writer.writerow([list_body_spam_cleaned[i], "1"])
    f.close()

with open("legit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_legit_cleaned)):
        writer.writerow([list_body_legit_cleaned[i], "0"])
    f.close()
####################################################

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
                    # returns a bytes object
                    strtext = payload.decode("latin-1")
                except:
                    strtext = part.get_payload(decode=False)
                    # returns a bytes object
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
            # email = striphtml(email)
            # email = email.replace("<", "").replace(">", "").replace("/n", "")
            # Normalize numbers
            e = re.sub("[0-9]+", "number", e)
            # Normalize URLs
            e = re.sub("(http|https)://[^\s]*", "httpAddress", e)
            # Normalize email addresses
            e = re.sub("[^\s]+@[^\s]+", "emailAddress", e)

            cleaned_email.append(e)
        except:
            # Normalize invalid emails
            if type(e) is list:
                e = ",".join(str(char) for char in e)
                cleaned_email.append(e)
    return cleaned_email
