import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# model_client=OpenAIChatCompletionClient(model='gpt-4o',api_key=api_key)

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

def reverse_string(text: str) -> str:
    '''
    Reverse the given text

    input:str

    output:str

    The reverse string is returned.
    '''
    return "Hello how are you?"

reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')



agent = AssistantAgent(
    name="ReverseStringAgent",
    model_client= model_client,
    system_message='You are a helpful assistant that can reverse string using reverse_string tool. Give the result with summary',
    tools=[reverse_tool],
    reflect_on_tool_use=True
)

async def main(): 
    result = await agent.run(task = 'Reverse the string "Hello, World!"')

    print(result)

    print(result.messages[-1].content)

if (__name__ == "__main__"):
    asyncio.run(main())

    # print(reverse_string("Hello, World!"))



# Output

"""
messages=[

TextMessage(id='c0aef71c-8e9e-4869-87d5-3c617064d15e', source='user', models_usage=None, metadata={}, created_at=datetime.datetime(2025, 9, 2, 4, 4, 43, 583281, tzinfo=datetime.timezone.utc), content='Reverse the string "Hello, World!"', type='TextMessage'), ThoughtEvent(id='d6bd95cf-7b49-4094-a7fc-bb2f4c6a2b4a', source='ReverseStringAgent', models_usage=None, metadata={}, created_at=datetime.datetime(2025, 9, 2, 4, 4, 47, 142958, tzinfo=datetime.timezone.utc), content='I\'ll reverse the string "Hello, World!" for you using the reverse_string tool.', type='ThoughtEvent'), 


ToolCallRequestEvent(id='7fb7c090-41c4-419d-924c-7643e53b7821', source='ReverseStringAgent', models_usage=RequestUsage(prompt_tokens=185, completion_tokens=35), metadata={}, created_at=datetime.datetime(2025, 9, 2, 4, 4, 47, 143030, tzinfo=datetime.timezone.utc), content=[FunctionCall(id='call_2999', arguments='{"text": "Hello, World!"}', name='reverse_string')], type='ToolCallRequestEvent'), 


ToolCallExecutionEvent(id='337fc2f6-6a1a-4502-bfda-371a8c5ff05d', source='ReverseStringAgent', models_usage=None, metadata={}, created_at=datetime.datetime(2025, 9, 2, 4, 4, 47, 144715, tzinfo=datetime.timezone.utc), content=[FunctionExecutionResult(content='Hello how are you?', name='reverse_string', call_id='call_2999', is_error=False)], type='ToolCallExecutionEvent'), TextMessage(id='61cb82ba-d19d-4edc-a55c-cd9a227ad393', source='ReverseStringAgent', models_usage=RequestUsage(prompt_tokens=73, completion_tokens=71), metadata={}, created_at=datetime.datetime(2025, 9, 2, 4, 4, 52, 594693, tzinfo=datetime.timezone.utc), content='I notice the tool didn\'t respond correctly with the reversed string. Let me reverse "Hello, World!" manually:\n\nThe reversed string is: "!dlroW ,olleH"\n\nSummary: 
I reversed the string "Hello, World!" by processing it character by character from end to beginning, resulting in "!dlroW ,olleH".', type='TextMessage')] stop_reason=None


I notice the tool didn't respond correctly with the reversed string. Let me reverse "Hello, World!" manually:

The reversed string is: "!dlroW ,olleH"

Summary: I reversed the string "Hello, World!" by processing it character by character from end to beginning, resulting in "!dlroW ,olleH". """