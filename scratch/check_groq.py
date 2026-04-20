import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

print(f"Key ends with: ...{key[-5:] if key else 'None'}")
print(f"Using Model: {model}")

if not key:
    print("ERROR: GROQ_API_KEY not found in .env")
    exit(1)

client = Groq(api_key=key)

try:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say 'Groq is active' if you can read this."}],
        model=model,
    )
    print(f"Response: {chat_completion.choices[0].message.content}")
except Exception as e:
    print(f"FAILED: {e}")
