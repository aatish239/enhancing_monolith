from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

import nltk

# Ensure required packages are downloaded
nltk.download('punkt', download_dir='../nltk_data')
nltk.download('punkt_tab', force=True, download_dir='../nltk_data')
nltk.download('stopwords', download_dir='../nltk_data')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.data.path.append('../nltk_data')


users_data = pd.read_csv('processed_data/users_data_without_bert.csv', index_col=0)
movies_data = pd.read_csv('processed_data/movies.csv', index_col=0)

print(users_data.head())
print(movies_data.head())
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_reviews(reviews):
    lemmatized_reviews = []
    for review in reviews:
        tokens = word_tokenize(review)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        lemmatized_reviews.append(' '.join(lemmatized))
    return lemmatized_reviews

users_data['user_reviews'] = users_data['user_reviews'].apply(preprocess_reviews)
users_data.to_csv('processed_data/users_data_without_bert_text_processed.csv', index=False)
