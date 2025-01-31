from textblob import TextBlob
import ast

import pandas as pd

# Load the dataset
file_path = "../../datasets/user_data.csv"
df = pd.read_csv(file_path)

# Display basic information and first few rows
df_info = df.info()
df_head = df.head()


# Function to parse list-like string entries
def parse_list(column):
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert string lists to actual lists
df["movie_ids"] = parse_list(df["movie_ids"])
df["user_ratings"] = parse_list(df["user_ratings"])
df["user_reviews"] = parse_list(df["user_reviews"])

# Feature extraction functions
def calculate_features(row):
    ratings = row["user_ratings"]
    reviews = row["user_reviews"]

    # Engagement Features
    num_movies_rated = len(ratings)
    avg_rating = sum(ratings) / num_movies_rated if num_movies_rated > 0 else 0
    rating_std = pd.Series(ratings).std() if num_movies_rated > 1 else 0

    # Behavioral Features
    high_rated_movies = sum(1 for r in ratings if r >= 8)
    low_rated_movies = sum(1 for r in ratings if r <= 4)

    # Review-Based Features
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in reviews]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    avg_review_length = sum(len(review.split()) for review in reviews) / len(reviews) if reviews else 0

    return pd.Series([
        num_movies_rated, avg_rating, rating_std,
        high_rated_movies, low_rated_movies,
        avg_sentiment, avg_review_length
    ])

# Apply feature extraction
df_features = df.apply(calculate_features, axis=1)
df_features.columns = [
    "num_movies_rated", "avg_rating", "rating_std",
    "high_rated_movies", "low_rated_movies",
    "avg_sentiment", "avg_review_length"
]

# Merge extracted features with user_id
df_final = pd.concat([df["user_id"], df_features], axis=1)
df_final.to_csv('../../datasets/user_features_non_bert.csv')