from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../.config/genai-experiments-397219-aaf888cfc4ad.json"


def get_service_account_credentials():
    service_account.Credentials.from_service_account_file(
        filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    # return credentials


def initialize_vertexai_params():
    get_service_account_credentials()
    vertexai.init(project="genai-experiments-397219", location="us-central1")
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.2
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    return model, parameters


def get_model_response(prompt_text, model, parameters):
    if model is None or parameters is None:
        model, parameters = initialize_vertexai_params()
    response = model.predict(prompt_text,
                             **parameters
                             )
    return response.text

