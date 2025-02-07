import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# ログの設定（ファイル名には実行時の時刻を付与）
logging.basicConfig(
    filename=f'logs/pot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# .envファイルの読み込み
load_dotenv()

# 環境変数から APIキー を取得
API_KEY = os.getenv("API_KEY")

# OpenAI クライアントの初期化
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def step1_knowledge_retrieval(question: str) -> str:
    """
    Step 1: ユーザーの質問に関連する知識をMECE原則に基づいて収集する
    """
    prompt = f"""
You are a capable assistant. Your task is to collect key knowledge relevant to the following question using the MECE (Mutually Exclusive and Collectively Exhaustive) principle.
Please output the information in the following bullet list format and limit your response to 5-7 key points.
- Category Name:
  - Item 1: Explanation including specific numbers or proper nouns (include reference such as URL or literature).
  - Item 2: Explanation (include reference).
- Category Name 2: ...
Do not provide any conclusions or in-depth reasoning at this stage.

[Question]
{question}
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
    Step 2: Step 1の知識をもとに論理的推論を行う
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

def step3_final_answer(question: str, knowledge_from_step1: str, reasoning_from_step2: str, answer_type: str) -> str:
    """
    Step 3: Step 1とStep 2の結果を統合して最終回答を導く。
    """
    prompt = f"""
You are a capable assistant. Using the knowledge and reasoning provided below, derive the final answer.
Summarize the information using the Pyramid Principle (Main Point, Sub Points, Supporting Data) and provide a concise final conclusion.
Please adhere to the following formatting rules:
- Write your final conclusion immediately after the header '### Conclusion' in a bullet list or short paragraph format.
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
    if answer_type == "boolean":
        prompt += "\nFor a boolean question, please answer with either 'Yes' or 'No' along with a brief supporting reason."

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

def step4_extract_final_answer(question: str, full_answer: str, answer_type: str) -> str:
    """
    Step 4: 詳細な回答から最終回答部分のみを抽出する。
    """
    # 基本のプロンプト
    prompt = f"""
You are provided with a complete answer that includes detailed reasoning and a final conclusion.
Extract and output only the final answer.
"""
    # answer_typeが"numerical"の場合のみ、数字に関するルールを追加
    if answer_type == "numerical":
        prompt += "All numbers must be written using Arabic numerals.\n"

    if answer_type == "boolean":
        prompt += "\nFor a boolean question, please answer with either 'Yes' or 'No'."
    
    prompt += f"""
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

def main(user_question: str, answer_type: str) -> str:
    logger.info(f"User's question:\n{user_question}\n\n")

    # Step 1: Knowledge retrieval
    knowledge = step1_knowledge_retrieval(user_question)

    # Step 2: Reasoning and organization
    reasoning = step2_reasoning(user_question, knowledge)

    # Step 3: Final answer (detailed, including reasoning)
    full_final_answer = step3_final_answer(user_question, knowledge, reasoning, answer_type)

    # Step 4: Extract only the final answer from the detailed answer
    final_answer = step4_extract_final_answer(user_question, full_final_answer, answer_type)
    return final_answer

if __name__ == "__main__":
    # 例: ユーザーの質問と回答タイプを指定して実行
    sample_question = "Which Republican candidate ran for president in 2008 but did not win presidential primaries?"
    answer_type = "entity"
    result = main(sample_question, answer_type)
    print(result)
