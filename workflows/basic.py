import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=API_KEY
)

chat_completion = client.chat.completions.create(
    messages = [
        {
            "role": "user",
            "content": "Explain what is Machine Learning."
        }
    ],
    model = "llama-3.3-70b-versatile"
)

print(chat_completion.choices[0].message.content)