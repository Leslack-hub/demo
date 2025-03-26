import sys

from openai import OpenAI


# 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
# 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
def RunOpenAi(model, content):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content.replace("\"", ""), end="")


models = {"1": "mistralai/mixtral-8x22b-instruct-v0.1", "2": "meta/llama3-70b-instruct", "3": "nvidia/nemotron-4-340b-instruct"}

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(0)
    modelKey = sys.argv[1]
    RunOpenAi(models[modelKey], " ".join(sys.argv[2:]))
    # RunOpenAi("meta/llama3-70b-instruct", "how to learn Deep learning")
