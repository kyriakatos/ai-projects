import anthropic

client = anthropic.Anthropic(api_key="CLAUDE_API_KEY")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(response.content[0].text)