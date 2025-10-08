import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_model_client():
    ## LLM
    load_dotenv()
    open_router_api_key = os.getenv('OPEN_ROUTER_API_KEY')

    model_client =  OpenAIChatCompletionClient(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat-v3.1:free",
        api_key = open_router_api_key,
        model_info={
            "family":'deepseek',
            "vision" :True,
            "function_calling":True,
            "json_output": False
        }
    )

    return model_client