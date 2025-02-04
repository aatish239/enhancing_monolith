# import statements
import pandas as pd

# loading the databases from non commercial imdb website
ratings = pd.read_csv("../raw_datasets/ratings.tsv", sep='\t')
crew = pd.read_csv('../raw_datasets/crew.tsv', sep='\t')
episode = pd.read_csv('../raw_datasets/episode.tsv', sep='\t')
basics = pd.read_csv('../raw_datasets/basics.tsv', sep='\t')
title_basics = pd.read_csv('../raw_datasets/title_basics.tsv', sep='\t', low_memory=False)

# merge the datasets to create unified movies dataset and movies people dataset
inter1 = pd.merge(ratings, crew, on='tconst', how='left')
movies_df = pd.merge(inter1, title_basics, on='tconst', how='left')

# export the datasets to the csv files
ratings.to_csv('../raw_datasets/movie_ratings.csv') # will be used for web scraping
movies_df.to_csv('../raw_datasets/movies.csv')
basics.to_csv('../datasets/movie_people.csv')

# print statements
print("Movies dataset head", movies_df.head())
print("Movies people head", basics.head())

movies_df = pd.read_csv('../raw_datasets/movies.csv', index_col=0)
users_df = pd.read_csv('../raw_datasets/moview_user_reviews.csv', names=['user_id', 'movie_id', 'user_rating',
                                                                       'user_review'])
movies_df_1 = movies_df[movies_df['tconst'].isin(users_df.movie_id)]

users_df = pd.read_csv('../raw_datasets/moview_user_reviews_20k.csv', names=['user_id', 'movie_id', 'user_rating',
                                                                    'user_review'])
movies_df_2 = movies_df[movies_df['tconst'].isin(users_df.movie_id)]
combined_df = pd.concat([movies_df_1, movies_df_2]).drop_duplicates(subset='tconst')
combined_df.to_csv('../datasets/movies.csv')