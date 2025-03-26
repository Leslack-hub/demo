from anthropic import AnthropicVertex

LOCATION = "us-east5"

client = AnthropicVertex(region=LOCATION, project_id="stable-victory-436002-c7")

message = client.messages.create(
  max_tokens=1024,
  messages=[
    {
      "role": "user",
      "content": "Send me a recipe for banana bread.",
    }
  ],
  model="claude-3-opus@20240229",
)
print(message.model_dump_json(indent=2))