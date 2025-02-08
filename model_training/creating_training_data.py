# Load and Analyze the User Data
# Feature Engineering:
# Merge user and movie data.
# Process genres, ratings, and BERT embeddings.
# Model Design:
# Neural network with user and movie embeddings.
# Auxiliary tasks if beneficial.
# Training & Prediction:
# Recommend 2 movies for new/test users.


import pandas as pd

# Load the uploaded movie and user datasets
movies_file_path = '../processed_data/training_movies_data.csv'
users_file_path = '../processed_data/training_users_data.csv'

# Load datasets
movies_df = pd.read_csv(movies_file_path, index_col=0)
users_df = pd.read_csv(users_file_path, index_col=0)

# Display basic information for both datasets
movies_info = movies_df.info()
users_info = users_df.info()

# Display first few rows to understand the structure
movies_sample = movies_df.head()
users_sample = users_df.head()

print(movies_info, users_info, movies_sample, users_sample)


# Dataset Overview:
# Movies Dataset :
# Identifiers: tconst, primaryTitle, originalTitle
# Ratings: averageRating, numVotes
# Metadata: directors, writers, titleType, isAdult, startYear, runtimeMinutes
# Genres: Encoded as genre1, genre2, genre3
# Users Dataset:
# Identifiers: user_id
# Interactions: movie_ids (list of watched movies), user_ratings (normalized), user_reviews (BERT tensor)


import ast
import numpy as np

# Convert string representations of lists and tensors to actual Python objects
users_df['movie_ids'] = users_df['movie_ids'].apply(ast.literal_eval)
users_df['user_ratings'] = users_df['user_ratings'].apply(ast.literal_eval)

# Convert BERT tensor strings to numpy arrays
def convert_tensor_to_array(tensor_str):
    # Remove 'tensor([' and '])', then convert to numpy array
    tensor_str = tensor_str.replace('tensor([', '').replace('])', '')
    tensor_values = np.fromstring(tensor_str, sep=',')
    return tensor_values

users_df['user_reviews'] = users_df['user_reviews'].apply(convert_tensor_to_array)

# Explode movie_ids and user_ratings for merging
exploded_users_df = users_df.explode(['movie_ids', 'user_ratings'])

# Merge with movie data
merged_df = exploded_users_df.merge(movies_df, left_on='movie_ids', right_on='tconst', how='left')
print(merged_df.info(), merged_df.head())
merged_df.drop(columns=['tconst'], inplace=True)
merged_df.to_csv('../processed_data/merged_data_for_training.csv')


