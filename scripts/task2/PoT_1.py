import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json

# ログの設定（ファイル名には実行時の時刻を付与）
logging.basicConfig(
    filename=f'logs/pot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # ログファイル名
    filemode='a',                   # 'a'は追記モード（既存のログに追記する）
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO              # INFOレベル以上をログに記録
)
logger = logging.getLogger(__name__)

# .envファイルの読み込み
load_dotenv()

# 環境変数から APIキー を取得
API_KEY = os.getenv("API_KEY")

# OpenAI クライアントの初期化
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def step1_knowledge_retrieval(question: str, context: list) -> str:
    """
    Step 1: ユーザーの質問に関連する知識をMECE原則に基づいて収集する。
    ・明示的に与えられた情報（context）を利用する。
    ・contextは、以下のような形式で与えられる:
    
    [
        [
          "タイトル1",
          [
            "説明文1",
            "説明文2",
            ...
          ]
        ],
        [
          "タイトル2",
          [
            "説明文1",
            "説明文2",
            ...
          ]
        ],
        ...
    ]
    """
    # context情報を整形して文字列に変換（JSON形式で見やすくする）
    context_str = json.dumps(context, ensure_ascii=False, indent=2)
    
    prompt = f"""
You are a capable assistant. Your task is to collect key knowledge relevant to the following question using the MECE (Mutually Exclusive and Collectively Exhaustive) principle.
In this step, you are explicitly provided with information via the "context" section below.
Please output the information in the following bullet list format and limit your response to key points:
- **Category Name**:
  - **Item 1**: Explanation including specific numbers or proper nouns.
  - **Item 2**: Explanation.
- **Category Name 2**: ...
Do not provide any conclusions or in-depth reasoning at this stage.

[Question]
{question}

[Context Provided]
{context_str}
"""
    # ログにプロンプトを出力
    logger.info("=== Step 1: Knowledge Retrieval ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content

    # ログにGPTのレスポンスを出力
    logger.info(f"Response from GPT (Step 1):\n{output_text}\n\n")
    return output_text

def step2_reasoning(question: str, knowledge_from_step1: str) -> str:
    """
    Step 2: Step 1の知識をもとに論理的推論を行う。
    ピラミッド原則（Main Point, Sub Points, Supporting Data）とChain of Thought（CoT）を用いて論理の展開過程を明示する。
    """
    prompt = f"""
You are a capable assistant. Based on the knowledge provided below, perform logical reasoning for the given question.
Organize your reasoning using the Pyramid Principle with the following hierarchical structure:
【Main Point】: [State the main point]
  →【Sub Point 1】: [Detail]
     -【Supporting Data 1】: [Evidence or reference]
  →【Sub Point 2】: [Detail]
     -【Supporting Data 2】: [Evidence or reference]
Clearly indicate the chain-of-thought process that connects the evidence to your reasoning.
Do not provide the final conclusion at this stage; focus solely on outlining the reasoning steps.

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}
"""
    # ログにプロンプトを出力
    logger.info("=== Step 2: Reasoning and Organization ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content

    # ログにGPTのレスポンスを出力
    logger.info(f"Response from GPT (Step 2):\n{output_text}\n\n")
    return output_text

def step3_final_answer(question: str, knowledge_from_step1: str, reasoning_from_step2: str) -> str:
    """
    Step 3: Step 1とStep 2の結果を統合して最終回答を導く。
    """
    prompt = f"""
You are a capable assistant. Using the knowledge and reasoning provided below, derive the final answer.
Summarize the information using the Pyramid Principle (Main Point, Sub Points, Supporting Data) and provide a concise final conclusion.
Please adhere to the following formatting rules:
Example format:
  ### Conclusion
  1. [Your final answer with key reasons]
  2. [Additional supporting statement if necessary]

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}

[Reasoning from Step 2]
{reasoning_from_step2}
"""
    # ログにプロンプトを出力
    logger.info("=== Step 3: Final Answer ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content

    # ログにGPTのレスポンスを出力
    logger.info(f"Response from GPT (Step 3):\n{output_text}\n\n")
    return output_text

def step4_extract_final_answer(question: str, full_answer: str) -> str:
    """
    Step 4: 詳細な回答から最終回答部分のみを抽出する。
    """
    prompt = f"""
You are provided with a complete answer that includes detailed reasoning and a final conclusion.
Extract and output only the final answer.

[Question]
{question}

[Complete Answer]
{full_answer}
"""
    # ログにプロンプトを出力
    logger.info("=== Step 4: Final Answer Extraction ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content

    # ログにGPTのレスポンスを出力
    logger.info(f"Response from GPT (Step 4):\n{output_text}\n\n")
    return output_text

def main(user_question: str, context: list) -> str:
    logger.info(f"User's question:\n{user_question}\n\n")
    logger.info(f"Context provided:\n{context}\n\n")

    # Step 1: Knowledge retrieval (context情報を利用)
    knowledge = step1_knowledge_retrieval(user_question, context)

    # Step 2: Reasoning and organization
    reasoning = step2_reasoning(user_question, knowledge)

    # Step 3: Final answer (detailed, including reasoning)
    full_final_answer = step3_final_answer(user_question, knowledge, reasoning)

    # Step 4: Extract only the final answer from the detailed answer
    final_answer = step4_extract_final_answer(user_question, full_final_answer)
    return final_answer

if __name__ == "__main__":
    # 例: ユーザーの質問とcontext情報を指定して実行
    sample_question = "Which Republican candidate ran for president in 2008 but did not win presidential primaries?"

    # context情報の例
    sample_context = [
        [
          "2008 United States presidential election",
          [
            "The 2008 United States presidential election was the 56th quadrennial presidential election, held on Tuesday, November 4, 2008.",
            "Republican candidate John McCain ran for president in 2008 but did not win the presidential primaries."
          ]
        ]
    ]
    
    result = main(sample_question, sample_context)
    print(result)
