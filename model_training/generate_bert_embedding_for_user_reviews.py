# Data Splitting
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import ast

users_data = pd.read_csv('../processed_data/users_data_without_bert.csv.csv', index_col=0)
# print(users_data.head())
# train_data, temp_data = train_test_split(users_data, test_size=0.3, random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# select only those movies which are part of the user dataset.

# users_data['movie_ids'] = users_data['movie_ids'].apply(ast.literal_eval)

# # Flatten the list of movie IDs
# all_movie_ids = [movie_id for sublist in users_data['movie_ids'] for movie_id in sublist]

# # Get unique movie IDs
# unique_movie_ids = list(set(all_movie_ids))
# movies_data_raw = pd.read_csv('../processed_data/movies.csv', index_col=0)
# movies_data_raw[movies_data_raw['tconst'].isin(unique_movie_ids)].to_csv('../processed_data/sample_movies.csv')
# movies_data_raw[~(movies_data_raw['tconst'].isin(unique_movie_ids))].to_csv('../processed_data/test_movies_data.csv')

# store the test data for movies(items) and users
# val_data.to_csv('../processed_data/sample_movies.csv')
# test_data.to_csv('../processed_data/test_users_data.csv')

# BERT Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(reviews):
    inputs = tokenizer(reviews, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
    # Average if multiple embeddings exist
    if cls_embeddings.shape[0] > 1:
        return torch.mean(cls_embeddings, dim=0)  # Shape: [768]
    else:
        return cls_embeddings.squeeze(0)  # Shape: [768]


# train_data['bert_embeddings'] = train_data['user_reviews'].apply(get_bert_embeddings)

# Iterating over rows
train_data_with_embeddings = pd.DataFrame(columns=users_data.columns)
train_data_with_embeddings.to_csv('../processed_data/users_data_bert_embeddings.csv', header=True)
batch_data = []
batch_size = 10
for index, row in users_data.iterrows():
    user_reviews = row['user_reviews']
    embedding = get_bert_embeddings(user_reviews)
    row_copy = row.copy()
    row_copy['user_reviews'] = embedding
    print(f"Index: {index}, row {row_copy}")
    batch_data.append(row_copy)
    if (index + 1) % batch_size == 0:
        batch_df = pd.DataFrame(batch_data)
        batch_df.to_csv('../processed_data/users_data_bert_embeddings.csv', mode='a', header=False)
        batch_data = []  # Clear memory

    
