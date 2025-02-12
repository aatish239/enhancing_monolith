import pandas as pd

df = pd.read_csv('../raw_datasets/moview_user_reviews.csv', names=['user_id', 'movie_id', 'user_rating', 'user_review'])
x = df.groupby('user_id')['movie_id'].apply(list).reset_index(name="movie_ids")
y = df.groupby('user_id')['user_rating'].apply(list).reset_index(name="user_ratings")
z = df.groupby('user_id')['user_review'].apply(list).reset_index(name="user_reviews")

user_database_1 = x.merge(y, on='user_id').merge(z, on='user_id')
user_database_1.to_csv('../raw_datasets/user_database_1.csv')

df = pd.read_csv('../raw_datasets/moview_user_reviews_20k.csv', names=['user_id', 'movie_id', 'user_rating',
                                                                       'user_review'])
x = df.groupby('user_id')['movie_id'].apply(list).reset_index(name="movie_ids")
y = df.groupby('user_id')['user_rating'].apply(list).reset_index(name="user_ratings")
z = df.groupby('user_id')['user_review'].apply(list).reset_index(name="user_reviews")

user_database_2 = x.merge(y, on='user_id').merge(z, on='user_id')
user_database_2.to_csv('../raw_datasets/user_database_2.csv')

df1 = pd.read_csv('../raw_datasets/user_database_1.csv', index_col=0)
df2 = pd.read_csv('../raw_datasets/user_database_2.csv', index_col=0)
combined_df = pd.concat([df1, df2]).drop_duplicates(subset='user_id')
combined_df.to_csv('../datasets/user_data.csv')
print(combined_df.head())
print(combined_df.shape)
