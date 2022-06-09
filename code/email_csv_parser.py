import email
import os
from base64 import decode
from email import policy
import csv
import splitfolders

# splitfolders.fixed(
#     r"Enron-Spam/",
#     output="src",
#     seed=42,
#     fixed=(500, 500),
#     oversample=False,
#     group_prefix=None,
# )

# Obtain the emails from the folder
listLegit = os.listdir(r"src/test/legítimo")
listSpam = os.listdir(r"src/test/no_deseado")

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


# Parsing the emails
list_body_legit = parse_email(listLegit, r"src/test/legítimo")
list_body_spam = parse_email(listSpam, r"src/test/no_deseado")

# Email cleaner function
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
            # Invalid email is ignored
            pass
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
    for i in list_body:
        try:
            list_body = i.replace("<", "").replace(">", "").replace("\n", "")
            cleaned_email.append(list_body)
        except:
            # Invalid email is ignored
            pass
    return cleaned_email
