import pandas as pd

# Load the data
file_path = '../datasets/user_features_non_bert.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data.info(), data.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

# 1. Examination of Rating Patterns
plt.figure(figsize=(10, 6))
sns.histplot(data['avg_rating'], bins=20, kde=True)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# 2. Review Sentiment Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_rating', y='avg_sentiment', data=data)
plt.title('Average Rating vs. Average Sentiment')
plt.xlabel('Average Rating')
plt.ylabel('Average Sentiment')
plt.show()

# 3. User Activity Trends
plt.figure(figsize=(10, 6))
sns.histplot(data['num_movies_rated'], bins=30, kde=True)
plt.title('Distribution of Number of Movies Rated per User')
plt.xlabel('Number of Movies Rated')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
correlation_matrix = data[['num_movies_rated', 'avg_rating', 'rating_std',
                            'high_rated_movies', 'low_rated_movies',
                            'avg_sentiment', 'avg_review_length']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of User Features')
plt.show()

