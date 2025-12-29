import os
import instructor

from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=API_KEY
)

client = instructor.from_groq(client) # for parsing


# Define response format in a Pydantic Model

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# Call Model

chat_completion = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    response_model = CalendarEvent,
    messages = [
        {
            "role": "system",
            "content": "Extract the event information."
        },
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday"
        }
    ]
)

# Parse response

event = chat_completion
event
event.name
event.date
event.participants