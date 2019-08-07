import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv('Spam_sms/spam.csv', encoding='ISO-8859-1')
df = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)

df.columns = ['labels', 'data']
print(df[: 2])

df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].as_matrix()

count_vectorizer = CountVectorizer(decode_error = 'ignore')
X = count_vectorizer.fit_transform(df['data'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

model = MultinomialNB()
model.fit(X_train, Y_train)
print("Train score: ", model.score(X_train, Y_train))
print("Test score: ", model.score(X_test, Y_test))

def visualization(label):
    words = ''
    for msg in df[df['labels'] ==label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width = 600, height = 400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
visualization('spam')
visualization('ham')


df['predictions'] = model.predict(X)

sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
    
    
not_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_spam:
    print(msg)

