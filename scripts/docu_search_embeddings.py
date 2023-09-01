import ast
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
import time
from vertex_ai import *


def generate_embeddings(text: str, model: vertexai.language_models):
    embeddings = model.get_embeddings([text])
    time.sleep(10)
    return embeddings[0].values


def compute_text_embeddings(restaurant_name: str, restaurants: pd.DataFrame,
                            model_name: Optional[str] = "textembedding-gecko@001") -> pd.DataFrame:
    model = get_embedding_model(model_name)
    generate_model_embeddings = partial(generate_embeddings, model=model)
    restaurants["embeddings"] = restaurants["review_text"].apply(generate_model_embeddings)
    restaurants.to_csv(f"../data/embeddings/{restaurant_name}_{model_name}_embeddings.csv", index=False)
    print("\t [Debugging] Embeddings computed & saved!")
    return restaurants


def load_embeddings_file(file_path: str):
    embeddings_df = pd.read_csv(file_path)
    return embeddings_df


def get_text_embeddings(restaurant_name: str, restaurants: pd.DataFrame, model_name: str) -> pd.DataFrame:
    file_name = f"{restaurant_name}_{model_name}_embeddings.csv"
    file_path = Path(f"{os.getcwd()}/../data/embeddings/{file_name}")

    if file_path.is_file():
        embeddings_df = load_embeddings_file(str(file_path))
    else:
        embeddings_df = compute_text_embeddings(restaurant_name, restaurants, model_name="textembedding-gecko@001")
    return embeddings_df


def find_best_match(prompt: str, restaurants: pd.DataFrame) -> str:
    model = get_embedding_model()
    query_embedding = generate_embeddings(prompt, model)

    restaurants["embeddings"] = restaurants["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    dot_products = np.dot(np.stack(restaurants['embeddings']), query_embedding)

    idx = np.argmax(dot_products)
    return restaurants.iloc[idx]["review_text"]  # Return text from index with max value


def get_final_response_embeddings(restaurant_name: str, prompt: str, restaurants: pd.DataFrame) -> str:
    print(f"Number of reviews to summarize: {len(restaurants)}")
    embeddings_df = get_text_embeddings(restaurant_name, restaurants, model_name="textembedding-gecko@001")
    final_response = find_best_match(prompt, embeddings_df)
    return final_response
