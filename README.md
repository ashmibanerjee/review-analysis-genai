# Yelp Reviews Analysis using Google Vertex AI

## Problem Statement & Motivation
Restaurant owners often want to know what their customers think about their restaurants. They can use this information to improve their menus, service, and atmosphere. However, it can be time-consuming and difficult to read through all the reviews that customers leave.

In this project, we'll teach developers how to accomplish this, by writing code with the `PaLM API`, and designing clever prompts.

## Solution
In this project, we use a sample dataset from the `Yelp Business reviews Dataset` to summarize reviews for the 
restaurants. 

The idea is that for a restaurant in the sample data, we will use Vertex AI `TextGenerationModel` to summarize the text according to the prompts.
The prompts can be either supplied by the user or generated randomly 
from a list of default prompts. 

However, we encountered several challenges in this process and this is how we overcame them.

## Challenges

### #1: Palm API and MakerSuite not available in the EU
The PaLM API and MakerSuite are not available for users in the EU. We were recommended to use VPN to access it, but this would have meant that our end-users in the EU would not be able to use it either. Instead, we chose to use `GCP VertexAI`.

Setting it up was challenging as we had to download the credentials.json for your Google Cloud Service Account and then access it through it.
```
credentials = service_account.Credentials.from_service_account_file(
                filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
              )
```
### #2: Exceeding input token limit
The main challenge that we encountered was that the Vertex AI `TextGenerationModel` has an input token limit. This means that we cannot feed all the reviews for a particular restaurant into the model at once.

To overcome this challenge, we broke the reviews up into smaller chunks and fed them into the model one at a time. We used the `tiktoken` library and the `gpt-3.5-turbo` model to estimate the number of tokens in each review. We also added a buffer to ensure that our reviews never exceeded the input token limit.

Once we had broken the reviews up into smaller chunks, we appended each chunk to the prompt statement and fed it into the Vertex AI `TextGenerationModel`. We then concatenated the responses from the model and fed them back into the model as a summary to generate the final response.

## Conclusion
We have successfully demonstrated how to use the Vertex AI `TextGenerationModel` to summarize reviews for restaurants. We have also overcome the challenges of using the PaLM API and exceeding the input token limit.

This project can be used by restaurant owners to gain insights into their customers' opinions about their restaurants. It can also be used by businesses to improve their products and services.

