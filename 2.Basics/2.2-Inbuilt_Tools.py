import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_ext.tools.http import HttpTool

# Load environment variables
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

# Create an HTTP tool for the httpbin API
# For cat facts (https://catfact.ninja/fact)

'''
{
  "fact": "Cats with long, lean bodies are more likely to be outgoing, and more protective and vocal than those with a stocky build.",
  "length": 121
}
'''


# Used ChatGPT to get the schema based on the base schema present in documentation
schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "A random cat fact"
            },
            "length": {
                "type": "integer",
                "description": "Length of the cat fact"
            }
        },
        "required": ["fact", "length"],
    }


http_tool = HttpTool(
    name="cat_facts_api",
    description="Get a cool cat fact",
    scheme="https",
    host="catfact.ninja",
    port=443,
    path="/fact",
    method="GET",
    return_type="json",
    json_schema=schema,
)

agent = AssistantAgent(
        name = "CatFactsAgent",
        model_client= model_client,
        system_message='You are a helpful assistant that can provide cat facts using the cat_facts_api tool. Give the result with summary',
        tools= [http_tool],
        reflect_on_tool_use= False
    )

async def main():
    result = await agent.run(task = 'Give me a random cat fact')

    print(result.messages)

if (__name__ == "__main__"):
    asyncio.run(main())