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

def generate_pot_prompt(question: str) -> str:
    """
    ユーザーの質問を元に、Pyramid of Thoughtを活用するためのプロンプトを生成する関数。
    """
    # ここでピラミッド原則 (Main Point, Sub Points, Supporting Data) と
    # Chain of Thought を意識した指示を与える
    pot_prompt = f"""
あなたは有能なアシスタントです。以下の手順で「ピラミッド型思考 (Pyramid of Thought)」を用いて質問に答えてください。

【質問】{question}

1. 知識の収集 (Pyramid Analysis)
   - 質問に関連する情報を整理し、MECE（漏れなくダブりなく）の原則でリストアップしてください。
   - 得られた知識を階層的に整理し、主要なポイントとサブポイントを構造化してください。

2. 論理的思考 (Chain of Thought)
   - 上記の知識から論理的に結論を導く思考過程を示してください。
   - メインポイントを明確化し、サブポイントや具体例を使って結論を裏付けてください。

3. 最終的な回答
   - 上記のプロセスを踏まえて、わかりやすく簡潔に最終的な回答を提示してください。

【出力フォーマット】
- ピラミッド構造に基づく知識整理（MECEを意識）
- 論理展開の要点
- 最終的な結論

以上の手順とフォーマットを守りながら、以下の質問に回答してください:
「{question}」
"""
    return pot_prompt.strip()

def main():
    # ユーザーから任意の質問を受け取る
    user_question = input("質問を入力してください: ")

    # PoTプロンプトを生成
    pot_prompt = generate_pot_prompt(user_question)

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
    print("\n===== 回答 =====")
    print(answer)

if __name__ == "__main__":
    main()
