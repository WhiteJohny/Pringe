import pandas as pd

from collaborative.user_based_cf import UserBasedCF
from utils.data_loader import load_movielens_1m
from baseline.popularity import PopularRecommender
from content_based.content_recommender import ContentRecommender
from collaborative.item_based_cf import ItemBasedCF
from two_tower import TwoTowerRecommender

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


def main():
    ratings, movies, users = load_movielens_1m()
    print(f"Loaded: {len(ratings)} ratings")

    pop = PopularRecommender(ratings, movies)
    baseline_df = pop.recommend(5)
    print("\nPOPULAR RECOMMENDER\n")
    print(baseline_df)

    content = ContentRecommender(movies)
    content_df = content.recommend(movie_id=1, top_k=5)
    print("\nCONTENT-BASED RECOMMENDER\n")
    print(content_df.to_string(index=False))

    cf_item = ItemBasedCF(ratings)
    ibcf_ids = list(map(int, cf_item.recommend(movie_id=1, top_k=5)))
    ibcf_df = movies[movies["movie_id"].isin(ibcf_ids)]
    print("\nITEM-BASED COLLABORATIVE RECOMMENDER\n")
    print(ibcf_df.to_string(index=False))

    cf_user = UserBasedCF(ratings)
    ubcf_ids = list(map(int, cf_user.recommend(user_id=1, top_k=5)))
    ubcf_df = movies[movies["movie_id"].isin(ubcf_ids)]
    print("\nUSER-BASED COLLABORATIVE RECOMMENDER\n")
    print(ubcf_df.to_string(index=False))

    two_tower = TwoTowerRecommender(users, movies)
    two_tower_df = two_tower.recommend(user_id=1, top_k=5)
    print("\nTWO-TOWER RECOMMENDER\n")
    print(two_tower_df.to_string(index=False))


if __name__ == "__main__":
    main()
