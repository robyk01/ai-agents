import json
import os
import requests
import instructor

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=API_KEY
)


# Define the tool that we want to call

def get_weather(latitude, longitude):
    """
    Public API for weather data
    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    data = response.json()
    return data["current"]


# Call model with the tool we defined

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


system_prompt = "You are a helpful weather assistant."

messages = [
    {
        "role": "user",
        "content": "What's the weather in Paris today?"
    },
    {
        "role": "system",
        "content": system_prompt
    }
]

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
)


# Model decides to call funtions

completion.model_dump()


# Execute get_weather function

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

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


# Supply result and call the model again

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The temperature in Celsius degrees."
    )

    response: str = Field(
        description="A natural language response to the user query."
    )


client = instructor.from_groq(client)

completion2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    tools=tools,
    messages=messages,
    response_model=WeatherResponse
)


# Check model response

final_result = completion2
final_result
final_result.temperature
final_result.response

