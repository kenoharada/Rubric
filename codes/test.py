from llm_router import get_llm_response

# model_name = "deepseek/deepseek-chat-v3.1"
# model_name = "meta-llama/llama-4-maverick"
# model_name = "qwen/qwen3-next-80b-a3b-instruct"
# params = {"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":8192,"provider":{"only":["OpenAI","Google"],"require_parameters":True}}
# model_name = "google/gemini-3-flash-preview"
# params = {"reasoning":{"effort":"high"},"max_tokens":8192,"provider":{"only":["OpenAI","Google"]}}
model_name = "openai/gpt-5-mini"
# まずは最小構成で疎通確認:
# - provider は azure 固定
# - require_parameters は provider 内に置く
# - reasoning は疎通確認後に追加する
# - gpt-5 系は max_tokens ではなく max_completion_tokens を優先
params = {
    # まずはパラメータを最小化して BYOK 疎通のみ確認
    "provider": {
        "only": ["azure"],
        "allow_fallbacks": False,
    },
}
response = get_llm_response(
    messages=[
        {"role": "user", "content": "好きな数字は？"}
    ],
    model_name=model_name,
    config=params,
    verbose=True,
)
print(response)

# import requests
# # List available models (GET /models)
# response = requests.get(
#   "https://openrouter.ai/api/v1/models",
#   headers={},
# )
# print(response.json())
# with open("models.json", "w") as f:
#     f.write(response.text)
