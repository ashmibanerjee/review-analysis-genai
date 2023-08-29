from typing import List, Optional
import random
import pandas as pd
import time
import tiktoken
from vertex_ai import get_model_response, initialize_vertexai_params

INPUT_TOKEN_LIMIT = 1000


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


def get_response_in_chunks(prompt: str, items: List[str]) -> str:
    print(f"Number of reviews to summarize: {len(items)}")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    model, parameters = initialize_vertexai_params()

    prompt_token_count = len(encoding.encode(prompt))
    token_count_sum = prompt_token_count

    generated_responses = []
    prompt_text_chunk = "Summarize in third person" + prompt

    for idx, item in enumerate(items):
        token_count = len(encoding.encode(item))
        token_count_sum += token_count
        if token_count_sum < 1000:
            prompt_text_chunk += " " + item
        else:
            print(f"\t [Debugging] Model requests upto idx:{idx}")
            chunk_response = get_model_response(prompt_text_chunk, model=model, parameters=parameters)
            generated_responses.append(chunk_response)
            prompt_text_chunk = "Summarize in third person" + prompt
            token_count_sum = prompt_token_count
            time.sleep(5)
    concatenated_responses = ''.join(generated_responses)
    if len(encoding.encode(concatenated_responses)) > INPUT_TOKEN_LIMIT:
        return get_response_in_chunks(prompt, generated_responses)
    # print(f"Tokens in generated_responses: {len(encoding.encode(concatenated_responses))}")
    return ''.join(generated_responses)


def get_final_response(prompt, items):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    model, parameters = initialize_vertexai_params()

    generated_response_chunks = get_response_in_chunks(prompt, items)
    # print(generated_response_chunks)

    final_prompt = prompt + " " + generated_response_chunks
    token_count = len(encoding.encode(final_prompt))

    if token_count < INPUT_TOKEN_LIMIT:
        final_response = get_model_response(final_prompt, model, parameters)
        if len(final_response) == 0:
            return generated_response_chunks
    else:
        final_response = "ERROR! Adjust input token count again"
    return final_response


def main(restaurant_name: Optional[str] = None, prompt: Optional[str] = None):
    if restaurant_name is None:
        restaurant_name = get_restaurant_name_from_data()
    print(f"Restaurant Name: {restaurant_name}")
    if prompt is None:
        prompt = get_default_prompt()
    print(f"Prompt: {prompt}")
    restaurants = read_data()
    filtered_restaurants = restaurants.loc[restaurants["name"] == restaurant_name]
    reviews = list(filtered_restaurants["review_text"])
    response = get_final_response(prompt, reviews)

    print(f"Final response:\n{response}")


if __name__ == '__main__':
    '''In this project, we use a sample from the Yelp Business reviews Dataset to summarize reviews for the 
    restaurants. The idea is that for a restaurant in the sample data, we will use Vertex AI TextGenerationModel to 
    summarize the text according to the prompts. The prompts can be either supplied by the user or generated randomly 
    from a list of default prompts. The challenge here is that we can't feed all the reviews together for a 
    particular restaurant into the Vertex AI model as it often exceeds the number of input tokens it can handle. To 
    overcome this problem, we use feed the model only chunks of texts ensuring that the token count is well below the 
    accepted limit. We use the tiktoken library and the gpt-3.5-turbo model to give us an estimation of the token 
    count of the reviews. Reviews are appended to the prompt statement as long as it does not exceed the accepted 
    input token limit and then fed into the Vertex AI TextGenerationModel for response. Responses from the model are 
    the concatenated and it is finally fed to the model as a summary to generate the final response. '''
    main()
