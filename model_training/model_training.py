# Feature Engineering:
#
# Process genres using one-hot encoding or embeddings.
# Normalize ratings and runtime.
# Use user_reviews as user preference vectors.
# Model Design:
#
# Neural Network with:
# User Embeddings: user_id, user_reviews
# Movie Embeddings: Genres, directors, etc.
# Auxiliary Tasks: (e.g., rating prediction)
# Training & Predictions:
#
# Recommend 2 movies per user (top-N recommendation).