from google import genai
from google.genai import types

# Passing the key explicitly (if not using environment variables)
client = genai.Client(api_key="GEMINI_API_KEY")

config = types.GenerateContentConfig(
    system_instruction="You are an elite enterprise data architect. Be concise and technical.",
    temperature=0.2,
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain the core difference between SQL and NoSQL databases in two sentences."
)

print(response.text)