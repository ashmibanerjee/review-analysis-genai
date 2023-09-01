from typing import Optional

from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../.config/genai-experiments-397219-aaf888cfc4ad.json"


def get_service_account_credentials():
    service_account.Credentials.from_service_account_file(
        filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    # return credentials


def initialize_vertexai_params(location: Optional[str] = "us-central1"):
    get_service_account_credentials()
    vertexai.init(project="genai-experiments-397219", location=location)
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.2
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    return model, parameters


def get_model_response(prompt_text, model, parameters) -> str:
    if model is None or parameters is None:
        model, parameters = initialize_vertexai_params()
    response = model.predict(prompt_text,
                             **parameters
                             )
    return response.text


def get_embedding_model(model_name: Optional[str] = "textembedding-gecko@001") -> vertexai.language_models:
    model = TextEmbeddingModel.from_pretrained(model_name)
    return model
