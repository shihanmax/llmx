from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9800/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def generate(query, client=client):
    chat_response = client.chat.completions.create(
        model="qwen2_5_14B_sft",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
    )

    resp = chat_response.choices[0].message.content
    return resp


def get_di(data, i):
    curr = data[i]
    instruct = curr["instruction"]
    input_ = curr["input"]
    output = curr["output"]

    print(f"instruct: {len(instruct)}, input: {len(input_)}")
    
    query = instruct + input_[-100000:]

    print(query)
    
    print(f"query: {len(query)}")
    return query, output
