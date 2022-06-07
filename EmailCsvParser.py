import email
import os
from base64 import decode
from email import policy
import csv

# Obtain the emails from the folder
listLegit = os.listdir(r"Enron-Spam/legítimo")
listSpam = os.listdir(r"Enron-Spam/no_deseado")

# Email parser function
def parse_email(list, path):
    # Create a list to store the emails
    list_body = []
    # Iterate through the emails
    for i in list:
        # read the email
        with open(path + "/" + i, encoding="latin-1") as f:
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

    return list_body


def clean_email(email):
    # remove html tags and \n from an email
    # create a list to store the cleaned email
    # remove invalid emails
    cleaned_email = []
    for i in email:
        try:
            email = i.replace("<", "").replace(">", "").replace("\n", "")

            cleaned_email.append(email)
        except:
            print()
    return cleaned_email


# Parsing the emails
list_body_legit = parse_email(listLegit, r"Enron-Spam/legítimo")
list_body_spam = parse_email(listSpam, r"Enron-Spam/no_deseado")
list_body_legit_cleaned = clean_email(list_body_legit)
list_body_spam_cleaned = clean_email(list_body_spam)

############### Creating CSV files ###################
rowHeaders = ["Email", "isSpam"]

with open("spam.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_spam_cleaned)):
        writer.writerow([list_body_spam_cleaned[i], "isSpam"])
    f.close()

with open("legit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(rowHeaders)
    for i in range(len(list_body_legit_cleaned)):
        writer.writerow([list_body_legit_cleaned[i], "isLegit"])
    f.close()
####################################################
