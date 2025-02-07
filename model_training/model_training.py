# Data Splitting
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

users_data = pd.read_csv('../processed_data/users_data.csv', index_col=0)
print(users_data.head())
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


# train_data['bert_embeddings'] = train_data['user_reviews'].apply(get_bert_embeddings)

# Iterating over rows
train_data_with_embeddings = pd.DataFrame(columns=train_data.columns)
train_data_with_embeddings.to_csv('../processed_data/training_users_data.csv', header=True)
for index, row in train_data.iterrows():
    user_reviews = row['user_reviews']
    embedding = get_bert_embeddings(user_reviews)
    row['user_reviews'] = embedding
    row_df = pd.DataFrame([row])
    print(f"Index: {index}, row {row}")
    row_df.to_csv('../processed_data/training_users_data.csv', mode='a', header=False)