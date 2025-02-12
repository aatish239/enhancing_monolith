import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
# Build the neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score,r2_score
import os

# Load the datasets
movies_df = pd.read_csv('processed_data/movies.csv')
users_df = pd.read_csv('processed_data/users_data_bert_embeddings.csv')

# movies_df = pd.read_csv('movies_sample_train (2).csv')
# users_df = pd.read_csv('users_sample_train (1).csv')

# Display the first few rows of each dataset to understand their structure
movies_df_head = movies_df.head()
users_df_head = users_df.head()

movies_df_shape = movies_df.shape
users_df_shape = users_df.shape

print(movies_df_head, users_df_head, movies_df_shape, users_df_shape)


# Convert string representations of lists to actual lists in users_df
users_df['movie_ids'] = users_df['movie_ids'].apply(ast.literal_eval)
users_df['user_ratings'] = users_df['user_ratings'].apply(ast.literal_eval)

# For 'user_reviews', extract the tensor values
def parse_tensor(tensor_str):
    # Remove 'tensor([' and '])' and split by comma
    values = tensor_str.replace('tensor([', '').replace('])', '').strip()
    return [float(x) for x in values.split(',') if x.strip()]

users_df['user_reviews'] = users_df['user_reviews'].apply(parse_tensor)

# Check the processed data
print(users_df.head())

# Function to split user_reviews if possible
def split_reviews(row):
    num_movies = len(row['movie_ids'])
    reviews = row['user_reviews']

    if len(reviews) % num_movies == 0:
        split_size = len(reviews) // num_movies
        return [reviews[i * split_size: (i + 1) * split_size] for i in range(num_movies)]
    else:
        return None  # Indicates problematic row

# Apply the function
users_df['split_reviews'] = users_df.apply(split_reviews, axis=1)

# Identify rows that couldn't be fixed
unfixable_rows = users_df[users_df['split_reviews'].isnull()]

# Remove unfixable rows and update user_reviews with split_reviews
users_df = users_df[users_df['split_reviews'].notnull()]
users_df['user_reviews'] = users_df['split_reviews']

# Drop the temporary column
users_df = users_df.drop(columns=['split_reviews'])

# Check the cleaned data
print(users_df.head())

# Explode the user dataframe to have one row per movie rating and review
users_exploded = users_df.explode(['movie_ids', 'user_ratings', 'user_reviews'])

# Merge with the movies dataset on movie_ids (tconst in movies_df)
merged_df = pd.merge(users_exploded, movies_df, left_on='movie_ids', right_on='tconst', how='inner')

# Function to pad tensors to length 768
def pad_tensor(tensor, target_length=768):
    if len(tensor) < target_length:
        return np.pad(tensor, (0, target_length - len(tensor)), 'constant')
    return tensor[:target_length]  # Truncate if longer than 768


# Combine genre columns into a list for one-hot encoding
merged_df['genres'] = merged_df[['genre1', 'genre2', 'genre3']].values.tolist()
merged_df['genres'] = merged_df['genres'].apply(lambda x: [g for g in x if pd.notnull(g)])

# One-hot encode genres
mlb_genres = MultiLabelBinarizer()
genre_encoded = mlb_genres.fit_transform(merged_df['genres'])

# Process directors and writers (split by comma and one-hot encode)
merged_df['directors_list'] = merged_df['directors'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
merged_df['writers_list'] = merged_df['writers'].apply(lambda x: x.split(',') if pd.notnull(x) else [])

mlb_directors = MultiLabelBinarizer()
mlb_writers = MultiLabelBinarizer()

directors_encoded = mlb_directors.fit_transform(merged_df['directors_list'])
writers_encoded = mlb_writers.fit_transform(merged_df['writers_list'])
# Apply padding
merged_df['user_reviews_padded'] = merged_df['user_reviews'].apply(pad_tensor)
merged_df.to_csv('../processed_data/merged_dataset.csv')

# Concatenate all features
X = np.concatenate([
    np.vstack(merged_df['user_reviews_padded'].values),  # Padded user review embeddings
    genre_encoded,
    directors_encoded,
    writers_encoded
], axis=1)

# Target variable
y = merged_df['user_ratings'].astype(float).values

# Check the shape of the feature matrix and target vector
print(X.shape, y.shape)
print(merged_df.head())
print(merged_df.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer
])

# Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))
# Save the original model
os.makedirs('model_directory', exist_ok=True)
model.save('model_directory/original_model.keras')

# Convert to a lightweight TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the lightweight model
with open('model_directory/lightweight_model.tflite', 'wb') as f:
    f.write(tflite_model)


# Evaluate
loss, mae = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Precision@K, Recall@K, Accuracy@K
K = 5
sorted_indices = np.argsort(y_pred.flatten())[::-1][:K]

y_true_top_k = (y_test[sorted_indices] >= 4).astype(int)  # Assuming rating >=4 is positive
y_pred_top_k = (y_pred.flatten()[sorted_indices] >= 4).astype(int)

precision_at_k = precision_score(y_true_top_k, y_pred_top_k, zero_division=0)
recall_at_k = recall_score(y_true_top_k, y_pred_top_k, zero_division=0)
accuracy_at_k = accuracy_score(y_true_top_k, y_pred_top_k)

print(f'Loss: {loss}, MAE: {mae}, R2: {r2}, rmse: {rmse}')
print(f'Precision@{K}: {precision_at_k}, Recall@{K}: {recall_at_k}, Accuracy@{K}: {accuracy_at_k}')

# Recommendation Function
def recommend_movies(fav_genres, fav_directors, fav_writers, top_n=5):
    genre_vector = mlb_genres.transform([fav_genres])
    director_vector = mlb_directors.transform([fav_directors])
    writer_vector = mlb_writers.transform([fav_writers])

    recommendations = []
    for index, row in merged_df.iterrows():
        features = np.concatenate([
            row['user_reviews_padded'],
            genre_vector[0],
            director_vector[0],
            writer_vector[0]
        ]).reshape(1, -1)
        predicted_rating = model.predict(features)[0][0]
        recommendations.append((row['primaryTitle'], predicted_rating))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# recommend movies along with user rating for a cold start user  based on genres, directors and writers.
# print the average user rating by that user.
# Example usage
fav_genres = ['Comedy', 'Drama', 'Action']
fav_directors = ['nm0412650', 'nm0000370']
fav_writers = ['nm0522871', 'nm0250873']

recommendations = recommend_movies(fav_genres, fav_directors, fav_writers)
for movie, score in recommendations:
    print(f"{movie}: {score}")
ratings = [rating for _, rating in recommendations]
print(np.mean(ratings))


def recommend_users(movie_genres, movie_directors, movie_writers, top_n=5):
    genre_vector = mlb_genres.transform([movie_genres])
    director_vector = mlb_directors.transform([movie_directors])
    writer_vector = mlb_writers.transform([movie_writers])

    recommendations = []
    for index, row in merged_df.iterrows():
        features = np.concatenate([
            row['user_reviews_padded'],
            genre_vector[0],
            director_vector[0],
            writer_vector[0]
        ]).reshape(1, -1)
        predicted_rating = model.predict(features)[0][0]
        recommendations.append((row['user_id'], predicted_rating))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# recommend users and user rating for the movie where the movie genre, writers and directors are provided.
movie_genres = ['Action', 'Thriller']
movie_directors = ['nm0883213']
movie_writers = ['nm0522871']

user_recommendations = recommend_users(movie_genres, movie_directors, movie_writers)
for user, score in user_recommendations:
    print(f"User {user}: {score}")
    
    
# Recommend Movies for a Test User
def evaluate_recommendations_for_test_user():
    # Select a user from the test data
    test_user_index = 0
    test_user_features = X_test[test_user_index]
    test_user_actual_rating = y_test[test_user_index]

    # Identify favorite genres, directors, and writers from the merged dataset
    test_user_data = merged_df.iloc[test_user_index]
    fav_genres = test_user_data['genres']
    fav_directors = test_user_data['directors_list']
    fav_writers = test_user_data['writers_list']

    # Get recommendations
    recommendations = recommend_movies(fav_genres, fav_directors, fav_writers)

    # Evaluate
    print(f"Actual Rating: {test_user_actual_rating}")
    for movie, predicted_rating in recommendations:
        print(f"Recommended Movie: {movie}, Predicted Rating: {predicted_rating}")

evaluate_recommendations_for_test_user()


def evaluate_lightweight_models():
    # Load the lightweight model for evaluation
    interpreter = tf.lite.Interpreter(model_path='model_directory/lightweight_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Function to predict using the lightweight model
    def predict_with_tflite(input_data):
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    # Evaluate both models
    def evaluate_model(y_true, y_pred, model_name="Model"):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        loss = mean_squared_error(y_true, y_pred)
        # accuracy = accuracy_score(y_true.round(), y_pred.round())
        # precision = precision_score(y_true.round(), y_pred.round(), average='weighted', zero_division=0)
        # recall = recall_score(y_true.round(), y_pred.round(), average='weighted', zero_division=0)
        # r2 = r2_score(y_true, y_pred)

        print(f"{model_name} Evaluation:\nRMSE: {rmse}, MAE: {mae}, Loss: {loss}\n")

    # Original Model Evaluation
    y_pred_original = model.predict(X_test)
    evaluate_model(y_test, y_pred_original, model_name="Original Model")

    # Lightweight Model Evaluation
    y_pred_tflite = np.array([predict_with_tflite(x.reshape(1, -1))[0][0] for x in X_test])
    evaluate_model(y_test, y_pred_tflite, model_name="Lightweight Model")
    print("lightweight" , os.stat('model_directory/lightweight_model.tflite').st_size)
    print("Original" , os.stat('model_directory/original_model.keras').st_size)
    
evaluate_lightweight_models()
    