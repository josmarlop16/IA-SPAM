from base64 import decode
from email import policy
import nltk
from nltk.corpus import words
import pandas as pd
import email
import os
import csv

from parso import parse

listSpam = os.listdir(r"Enron-Spam/no_deseado")

# Email parser function
def parse_email(list):
    # Create a list to store the emails
    list_body = []
    # Iterate through the emails
    for i in list:
        # Open the email
        with open(r"Enron-Spam/no_deseado/" + i , encoding="latin-1") as f:
            msg = email.message_from_file(f)
            if msg.is_multipart():
                for part in msg.walk():
                    try:
                        payload = part.get_payload(
                            decode=True
                        ) 
                        # returns a bytes object
                        strtext = payload.decode("latin-1")
                    except:
                        strtext = part.get_payload(
                            decode=False
                        ) 
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

list_email = parse_email(listSpam)

# delete html tags and \n from an email
def clean_email(email):
    # create a list to store the cleaned email
    cleaned_email = []
    for i in email:
        try:
            text = i.replace("<", "").replace(">", "").replace("\n", "")
            cleaned_email.append(text)
        except:
            print(i)
    # return the cleaned email
    return cleaned_email

clean = clean_email(list_email)



# save the cleaned emails to a csv file
with open("cleaned_emails.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for i in range(len(clean)):
        writer.writerow([clean[i]])
    f.close()


