import os
import json
import instructor

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=API_KEY
)

# Define the search knowledge base retrieval function

KB_PATH = Path(__file__).resolve().parent / "kb.json"
def search_kb(question: str):
    """
        Load the whole knowledge base from the JSON file.
    """
    if not KB_PATH.exists():
        return {"error": f"kb.json not found at {KB_PATH}"}
    with KB_PATH.open("r") as f:
        return json.load(f)
    


# Call model with search_kb tool defined

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."
    },
    {
        "role": "user",
        "content": "What is the return policy"
    }
]

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
)



# Model decides to call function(s)

completion.model_dump()



# Execute search_kb function

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    messages.append(completion.choices[0].message.model_dump(
        include={"role", "content", "tool_calls"}
    ))

    result = call_function(name, args)

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        }
    )

client = instructor.from_groq(client)


# Supply result and call model again

class KBSearch(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")

completions2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,
    response_model=KBSearch
)


# Chat Simulation

msg = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."
    },
]

client = Groq(
    api_key=API_KEY
)

print("--- Customer Support AI Agent ---")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    msg.append({"role": "user", "content": user_input})

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msg,
        tools=tools,
    )

    if chat.choices[0].message.tool_calls:
        for tool_call in chat.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = call_function(name, args)

            msg.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                    "content": "None",
                }
            )
        
            msg.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            ) 

            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=msg,
            )

    print("AI: ", chat.choices[0].message.content)
    msg.append({"role": "assistant", "content": chat.choices[0].message.content})



