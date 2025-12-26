from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

def save_research_note(filename: str, content: str) -> str:
    """
    Saves a research note or thought to a local file
    """

    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Succesfully saved note to {filename}"
    except Exception as err:
        return f"Error: {str(err)}"
        
        

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# system instruction
SYSTEM_PROMPT = """
You are a Research Assistant.
For every question the user asks, you must follow this exact format:

THOUGHT: (Describe your reasoning process and what you need to find out)
ANSWER: (Provide the final, concise response to the user)

If you don't know the answer, say so in the THOUGHT section.
"""

# create client
client = genai.Client(api_key=API_KEY)

# search tool
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# chat
chat = client.chats.create(
        model='gemini-2.5-flash-lite',
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=500,
            tools=[
                grounding_tool
            ]
        )
)

print("--- AI Agent Initialized ---")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = chat.send_message(user_input)

    print(f"AI: {response.text}\n")

    tokens = response.usage_metadata.total_token_count
    print(f"--- [Tokens Used: {tokens}] ---\n")