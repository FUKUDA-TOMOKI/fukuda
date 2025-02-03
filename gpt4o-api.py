import os
from dotenv import load_dotenv

# .env ファイルの内容を読み込む
load_dotenv()

# 環境変数から API キーを取得
API_KEY = os.getenv("API_KEY")

from openai import OpenAI

client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは！"}
    ]
)

print(response.choices[0].message.content)