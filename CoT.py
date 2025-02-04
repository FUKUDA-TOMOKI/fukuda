import os
from dotenv import load_dotenv
from openai import OpenAI

# .env ファイルの内容を読み込む
load_dotenv()

# 環境変数から API キーを取得
API_KEY = os.getenv("API_KEY")

# OpenAIクライアントを初期化
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def main(user_question: str) -> str:
    # ユーザーから任意の質問を受け取る
    # user_question = input("質問を入力してください: ")

    # PoTプロンプトを生成
    pot_prompt = (
        user_question +
        "\nLet's think step by step." +
        "\nPlease write your final conclusion immediately after the '### Conclusion' section header." +
        "\nPlease use Arabic numerals (e.g., 1, 2, 3) when writing numbers, rather than spelling them out with alphabetic characters."
        "\nPlease answer with either 'Yes' or 'No' when the question can clearly be answered using one of those options."
    )

    # ChatCompletion APIにリクエストを送信
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pot_prompt},
        ]
    )

    # 応答を表示
    answer = response.choices[0].message.content
    # print("\n===== 回答 =====")
    # print(answer)
    return answer

if __name__ == "__main__":
    main()
