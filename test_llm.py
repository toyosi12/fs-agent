from openai import OpenAI

# Note: The base_url varies by region. The following example uses the base_url for the Singapore region.
# - Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# - China (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1
client = OpenAI(
    api_key="your_api_key_here", 
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1"
)
completion = client.chat.completions.create(
    model="qwen3-coder-480b-a35b-instruct",
    messages=[{"role": "user", "content": "Who are you?"}]
)
print(completion.choices[0].message.content)