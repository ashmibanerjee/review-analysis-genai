import time
from typing import List
import tiktoken
from vertex_ai import get_model_response, initialize_vertexai_params

INPUT_TOKEN_LIMIT = 1000


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


def get_final_response_prompt_engg(prompt, items):
    """In this project, we use a sample from the Yelp Business reviews Dataset to summarize reviews for the
    restaurants. The idea is that for a restaurant in the sample data, we will use Vertex AI TextGenerationModel to
    summarize the text according to the prompts. The prompts can be either supplied by the user or generated randomly
    from a list of default prompts. The challenge here is that we can't feed all the reviews together for a
    particular restaurant into the Vertex AI model as it often exceeds the number of input tokens it can handle. To
    overcome this problem, we use feed the model only chunks of texts ensuring that the token count is well below the
    accepted limit. We use the tiktoken library and the gpt-3.5-turbo model to give us an estimation of the token
    count of the reviews. Reviews are appended to the prompt statement as long as it does not exceed the accepted
    input token limit and then fed into the Vertex AI TextGenerationModel for response. Responses from the model are
    the concatenated, and it is finally fed to the model as a summary to generate the final response. """

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
