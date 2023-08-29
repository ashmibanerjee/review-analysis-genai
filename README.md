# Yelp Reviews Analysis using Google Vertex AI

## Problem
Imagine you own a restaurant with hundreds of reviews. Can you use the PaLM API to answer questions like:
* What are the most popular menu items? 
* What do customers like most about my restaurant? 
* What can the restaurant improve? 

In this project, you'll teach developers how to accomplish this, by writing code with the `PaLM API`, and designing clever prompts.

## Solution
In this project, we use a sample from the `Yelp Business reviews Dataset` to summarize reviews for the 
restaurants. 

The idea is that for a restaurant in the sample data, we will use Vertex AI `TextGenerationModel` to summarize the text according to the prompts.
The prompts can be either supplied by the user or generated randomly 
from a list of default prompts. 

However, we encountered several challenges in this process and this is how we overcame them.

## Challenges

### Challenges 1: Palm API and MakerSuite not available in the EU
Palm API and MakerSuite not available not available for users in the EU. We were recommended to use VPN to access it. However, it comes with the disadvantage that our end-users in the EU won't be able to use it either. Hence, I chose to use `GCP VertexAI` instead.

Setting it up was challenging as we had to download the credentials.json for your Google Cloud Service Account and then access it through it.
```
credentials = service_account.Credentials.from_service_account_file(
                filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
              )
```
### Challenge 2: Exceeding input token limit
Even though the problem sounds straight forward, the challenge here is that we can't feed all the reviews together for a particular restaurant into the Vertex AI model as it often exceeds the number of `input tokens` it can handle. 

To overcome this problem, we use feed the model only chunks of texts ensuring that the token count is well below the accepted limit.

We use the `tiktoken` library and the `gpt-3.5-turbo` model to give us an estimation of the token count of the reviews. We also keep some buffer here to ensure that our reviews never run over the `input tokens` limit as it is not super accurate for `PalmAPI`/`Vertex AI`.

Reviews are appended to the prompt statement as long as it does not exceed the accepted input token limit and then fed into the Vertex AI ` TextGenerationModel` for response. 

Responses from the model are the concatenated and it is finally fed to the model as a summary to generate the final response.

