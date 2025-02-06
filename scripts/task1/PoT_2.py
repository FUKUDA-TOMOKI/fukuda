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

def step1_knowledge_retrieval(question: str) -> str:
    """
    ステップ1: 質問に対して関連する知識を収集し、MECEの原則に基づいて整理するようモデルに促す。
    """
    prompt = f"""
あなたは有能なアシスタントです。ユーザーの質問に対する知識をまず収集してください。
MECE（漏れなくダブりなく）の原則を使って情報を分類し、主要な項目をリストアップしてください。

【質問】{question}

指示：
- 質問に関連する重要な事実やデータを収集し、リスト化してください。
- 情報の整理段階なので、結論や推論はまだ行わなくてよいです。
- 必要に応じて数字や具体的な固有名詞を含めてください。
- 情報を提示する際には、必ずその根拠や出典を示してください。
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def step2_reasoning(question: str, knowledge_from_step1: str) -> str:
    """
    ステップ2: ステップ1で得られた知識を基に、論理的思考を行う。
    ここではピラミッド原則 (Main Point, Sub Points, Supporting Data) と
    Chain of Thought (CoT) を活用し、結論に至るまでの推論を組み立てる。
    """
    prompt = f"""
あなたは有能なアシスタントです。以下の情報をもとに、質問に対する推論を行ってください。

【質問】
{question}

【ステップ1で収集した知識】
{knowledge_from_step1}

指示：
1. ステップ1で得た知識を参照し、メインポイントとサブポイントを階層的に整理してください。
2. その情報から論理的に考え、結論を出す手前までの推論（Chain of Thought）を行ってください。
3. ここではまだ最終結論は示さず、「どんな根拠があるか」「どのように比較・推論を進めるか」など、
   ピラミッド原則でいう「サブポイント」や「Supporting Data」を洗い出してください。
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def step3_final_answer(question: str, knowledge_from_step1: str, reasoning_from_step2: str) -> str:
    """
    ステップ3: ステップ1とステップ2の結果を統合し、最終的な回答を出す。
    """
    prompt = f"""
あなたは有能なアシスタントです。以下の情報を踏まえて、最終的な回答を導いてください。

【質問】
{question}

【ステップ1で収集した知識】
{knowledge_from_step1}

【ステップ2の推論内容】
{reasoning_from_step2}

指示：
- これまで整理した知識と推論を総合して、ピラミッド原則（Main Point, Sub Points, Supporting Data）に基づく結論を示してください。
- 最終的な答えを簡潔に提示しつつ、そこに至る理由づけも要点として示してください。
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def main():
    # ユーザーから任意の質問を受け取る
    user_question = input("質問を入力してください: ")

    # ステップ1: 知識の収集
    knowledge = step1_knowledge_retrieval(user_question)
    print("\n----- ステップ1: 知識の収集結果 -----")
    print(knowledge)

    # ステップ2: 推論と整理
    reasoning = step2_reasoning(user_question, knowledge)
    print("\n----- ステップ2: 推論と整理 -----")
    print(reasoning)

    # ステップ3: 最終回答
    final_answer = step3_final_answer(user_question, knowledge, reasoning)
    print("\n===== 最終回答 =====")
    print(final_answer)

if __name__ == "__main__":
    main()
