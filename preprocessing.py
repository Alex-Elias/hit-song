import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer

df = pd.read_csv("charts.csv")

#filter region and chart
df = df[df['region'] == 'United States']
df = df[df['chart'] == 'top200']

#add column next to artist column to note number of artists in each song
df.insert(4, 'artist count', 1)
df['artist count'] = df['artist'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

#label encoding for chart and trend
trend_encoder = LabelEncoder()

df['trend_encoded'] = trend_encoder.fit_transform(df['trend'])

#stemming for title and artist
stemmer = PorterStemmer()

def stem_text(text):
    if isinstance(text, str):
        stemmed_words = [stemmer.stem(word) for word in text.split()]
        return ' '.join(stemmed_words)
    return ''

df['title'] = df['title'].apply(stem_text)

df.to_csv("processed_charts.csv", index=False)

df2 = pd.read_csv('processed_charts.csv')
print(df2)
