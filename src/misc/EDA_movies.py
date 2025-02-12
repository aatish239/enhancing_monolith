import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
movies_file_path = '../datasets/movies.csv'
movies_data = pd.read_csv(movies_file_path)

# Display basic information about the dataset
movies_data.info(), movies_data.head()

# Clean and preprocess the data
# Convert 'runtimeMinutes' to numeric, coercing errors to handle non-numeric values
movies_data['runtimeMinutes'] = pd.to_numeric(movies_data['runtimeMinutes'], errors='coerce')

# 1. Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(movies_data['averageRating'], bins=20, kde=True)
plt.title('Distribution of Average Movie Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# 2. Genre Popularity
# Explode genres into separate rows
genre_data = movies_data.assign(genres=movies_data['genres'].str.split(',')).explode('genres')
genre_counts = genre_data['genres'].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(y=genre_counts.index[:10], x=genre_counts.values[:10])
plt.title('Top 10 Most Popular Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# 3. Average Runtime
plt.figure(figsize=(10, 6))
sns.histplot(movies_data['runtimeMinutes'].dropna(), bins=30, kde=True)
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Frequency')
plt.show()

# 4. Year-wise Distribution of Movies
plt.figure(figsize=(12, 6))
sns.histplot(movies_data['startYear'], bins=50, kde=False)
plt.title('Year-wise Distribution of Movies')
plt.xlabel('Start Year')
plt.ylabel('Number of Movies')
plt.show()
