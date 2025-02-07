import pandas as pd
import ast

movies_file_path = 'drive/MyDrive/MSProject/datasets/movies.csv'
movies_data_raw = pd.read_csv(movies_file_path, index_col=0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
users_file_path = 'drive/MyDrive/MSProject/datasets/user_data.csv'
users_data_raw = pd.read_csv(users_file_path, index_col=0)


users_data = users_data_raw

users_data['movie_ids'] = users_data['movie_ids'].apply(ast.literal_eval)

# Flatten the list of movie IDs
all_movie_ids = [movie_id for sublist in users_data['movie_ids'] for movie_id in sublist]

# Get unique movie IDs
unique_movie_ids = list(set(all_movie_ids))
movies_data_sample = movies_data_raw[ movies_data_raw['tconst'].isin(unique_movie_ids)]
movies_data_sample.to_csv('drive/MyDrive/MSProject/finaldatasets/movies_data.csv')
users_data_raw.to_csv('drive/MyDrive/MSProject/finaldatasets/users_data.csv')

import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure required packages are downloaded
nltk.download('punkt', force=True, download_dir='drive/MyDrive/MSProject/nltk_data')
nltk.download('punkt_tab', force=True, download_dir='drive/MyDrive/MSProject/nltk_data')
nltk.download('stopwords', download_dir='drive/MyDrive/MSProject/nltk_data')
nltk.data.path.append('drive/MyDrive/MSProject/nltk_data')

# Data Loading
users_data = pd.read_csv('drive/MyDrive/MSProject/finaldatasets/users_data.csv')
movies_data = pd.read_csv('drive/MyDrive/MSProject/finaldatasets/movies_data.csv')

# Data Cleaning
movies_data.replace('\\N', np.nan, inplace=True)
movies_data['runtimeMinutes'] = pd.to_numeric(movies_data['runtimeMinutes'], errors='coerce')
movies_data['runtimeMinutes'].fillna(movies_data['runtimeMinutes'].median(), inplace=True)
movies_data['directors'].fillna('Unknown', inplace=True)
movies_data['writers'].fillna('Unknown', inplace=True)

# Convert stringified lists to lists
users_data['movie_ids'] = users_data['movie_ids'].apply(ast.literal_eval)
users_data['user_ratings'] = users_data['user_ratings'].apply(ast.literal_eval)
users_data['user_reviews'] = users_data['user_reviews'].apply(ast.literal_eval)

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

# Text Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_reviews(reviews):
    processed = []
    for review in reviews:
        tokens = word_tokenize(review)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        processed.append(' '.join(tokens))
    return processed

users_data['processed_reviews'] = users_data['user_reviews'].apply(preprocess_reviews)

# Data Splitting
train_data, temp_data = train_test_split(users_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

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

train_data['bert_embeddings'] = train_data['processed_reviews'].apply(get_bert_embeddings)

# Encoding Categorical Data
encoder = LabelEncoder()
train_data['user_id_enc'] = encoder.fit_transform(train_data['user_id'])

# Neural Network (MLP)
class RecommenderMLP(nn.Module):
    def __init__(self, input_dim):
        super(RecommenderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))

# Model Training
input_dim = train_data['bert_embeddings'].iloc[0].shape[0]
model = RecommenderMLP(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to tensors
print([embedding.shape for embedding in train_data['bert_embeddings']])

# Flatten the ratings as before
y_train = torch.tensor([rating for sublist in train_data['user_ratings'] for rating in sublist], dtype=torch.float32)

# Repeat embeddings for each rating
X_train = torch.cat([
    embedding.unsqueeze(0).repeat(len(ratings), 1)
    for embedding, ratings in zip(train_data['bert_embeddings'], train_data['user_ratings'])
])
print(f"X_train shape: {X_train.shape}")  # Should match the number of ratings
print(f"y_train shape: {y_train.shape}")  # Same as X_train[0]

# Training Loop
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Predictions
y_pred = model(X_train).detach().numpy()

# Metrics
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f'RMSE: {rmse}')

# Explainability with SHAP
import shap

# SHAP DeepExplainer
explainer = shap.DeepExplainer(model, X_train)

# Compute SHAP values
shap_values = explainer.shap_values(X_train)

# Summary Plot
shap.summary_plot(shap_values, features=X_train.numpy())

# explainer = shap.KernelExplainer(model, X_train[:50])  # Use a sample for faster computation
# shap_values = explainer.shap_values(X_train[:10])      # Explain on a smaller subset

# shap.summary_plot(shap_values, X_train[:10].numpy())

# Save Model
torch.save(model.state_dict(), 'recommender_model.pth')