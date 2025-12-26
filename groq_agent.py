import os
import json
import math
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

def calculate(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, safe_env)
        return str(result)
    except Exception as e:
        return f"Math Error: {str(e)}"

SYSTEM_PROMPT = "You are a calculator assistant. Use the calculate tool for every math problem."

tools = [
    {
        "type": "function", 
        "function": {
            "name": "calculate", 
            "description": "Calculate a math expression", 
            "parameters": {
                "type": "object", 
                "properties": {"expression": {"type": "string"}}, 
                "required": ["expression"]
            }
        }
    }
]

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"--- Groq Math Agent ({MODEL}) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]: break
    
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto" 
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"Agent is Acting: ({name} inputs: {args})...")

            result = calculate(**args)
            print(f"DEBUG: Tool Result -> {result}")

            messages.append({
                "role": "tool", 
                "tool_call_id": tool_call.id,
                "content": result
            })

        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="none"
        )
        
        print(f"\nAI: {final.choices[0].message.content}\n")
        messages.append(final.choices[0].message)
    else:
        print(f"AI: {msg.content}\n")
        messages.append(msg)