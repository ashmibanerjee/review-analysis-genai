from typing import Optional
import random
import pandas as pd
from prompt_engg import get_final_response_prompt_engg
from docu_search_embeddings import get_final_response_embeddings


def read_data():
    restaurants = pd.read_csv(
        "../data/yelp_academic_dataset_businesses_reviews_sample.csv")
    return restaurants


def get_restaurant_name_from_data():
    restaurants = read_data()
    restaurant_names = list(restaurants.name.unique())
    return random.choice(restaurant_names)


def get_default_prompt():
    default_prompts = ["What are the most popular menu items:", "What do customers like most about the restaurant:",
                       "What can the restaurant improve:", "What are the most disliked things about the restaurant:"]
    return random.choice(default_prompts)


def main(restaurant_name: Optional[str] = None, prompt: Optional[str] = None, approach: Optional[str] = "prompt"):
    if restaurant_name is None:
        restaurant_name = get_restaurant_name_from_data()
    print(f"Restaurant Name: {restaurant_name}")
    if prompt is None:
        prompt = get_default_prompt()
    print(f"Prompt: {prompt}")
    restaurants = read_data()
    filtered_restaurants = restaurants.loc[restaurants["name"] == restaurant_name]

    if approach == 1:
        reviews = list(filtered_restaurants["review_text"])
        response = get_final_response_prompt_engg(prompt, reviews)
    else:
        response = get_final_response_embeddings(restaurant_name, prompt, filtered_restaurants)

    print(f"Final response:\n{response}")


if __name__ == '__main__':
    main(approach="embedding")
