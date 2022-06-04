import nltk
import pandas as pd
# read training data & test data
df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("test.csv")
# read stopwords
stopwords = nltk.corpus.stopwords.words('english')