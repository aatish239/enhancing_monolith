import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def safe_literal_eval(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]  # Wrap single strings in a list
    return value  # If already a list, return as is

# Data Loading
users_data = pd.read_csv('../datasets/user_data.csv')
movies_data = pd.read_csv('../datasets/movies.csv')

# Data Cleaning
movies_data.replace('\\N', np.nan, inplace=True)
movies_data['runtimeMinutes'] = pd.to_numeric(movies_data['runtimeMinutes'], errors='coerce')
movies_data['runtimeMinutes'].fillna(movies_data['runtimeMinutes'].median(), inplace=True)
movies_data['directors'].fillna('Unknown', inplace=True)
movies_data['writers'].fillna('Unknown', inplace=True)

# Convert stringified lists to lists
users_data['movie_ids'] = users_data['movie_ids'].apply(safe_literal_eval)
users_data['user_ratings'] = users_data['user_ratings'].apply(safe_literal_eval)
users_data['user_reviews'] = users_data['user_reviews'].apply(safe_literal_eval)

# Convert stringified lists to lists
users_data['movie_ids'] = users_data['movie_ids'].apply(safe_literal_eval)
users_data['user_ratings'] = users_data['user_ratings'].apply(safe_literal_eval)
users_data['user_reviews'] = users_data['user_reviews'].apply(safe_literal_eval)

# Normalize Ratings
scaler = MinMaxScaler()
all_ratings = [rating for sublist in users_data['user_ratings'] for rating in sublist]
scaled_ratings = scaler.fit_transform(np.array(all_ratings).reshape(-1, 1)).flatten()
# Apply normalized ratings
index = 0
for i in range(len(users_data['user_ratings'])):
    length = len(users_data['user_ratings'][i])
    users_data.at[i, 'user_ratings'] = scaled_ratings[index:index+length].tolist()
    index += length
    
movies_data['averageRating'] = scaler.fit_transform(movies_data[['averageRating']])
label_encoder = LabelEncoder()
movies_data['titleType'] = label_encoder.fit_transform(movies_data['titleType'])
genres_split = movies_data['genres'].str.split(',', expand=True)
movies_data['genre1'] = genres_split[0]
movies_data['genre2'] = genres_split[1]
movies_data['genre3'] = genres_split[2]
movies_data.drop(columns=['genres'], inplace=True)
movies_data.replace('\\N', np.nan, inplace=True)
genre1 = movies_data['genre1'].to_list()
genre2 = movies_data['genre2'].to_list()
genre3 = movies_data['genre3'].to_list()
genre1.extend(genre2)
genre1.extend(genre3)
genre1 = set(genre1)

label_to_value = {}
for index, genre in enumerate(genre1):
  label_to_value.update({genre: index})

label_to_value[np.NaN] = label_to_value[None]

movies_data['genre1'] = movies_data['genre1'].map(label_to_value)
movies_data['genre2'] = movies_data['genre2'].map(label_to_value)
movies_data['genre3'] = movies_data['genre3'].map(label_to_value)

movies_data.to_csv('../processed_data/movies.csv')
users_data.to_csv('../processed_data/users_data.csv')


