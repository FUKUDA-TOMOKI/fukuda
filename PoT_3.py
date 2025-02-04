import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# ログの設定
logging.basicConfig(
    filename='app.log',            # ログファイル名
    filemode='a',                  # 'a'は追記モード（既存のログに追記する）
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO             # INFOレベル以上をログに記録
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
    Step 1: Collect knowledge related to the question according to the MECE principle.
    """
    prompt = f"""
You are a capable assistant. First, collect knowledge relevant to the user's question.
Use the MECE (Mutually Exclusive and Collectively Exhaustive) principle to categorize the information,
and list the main points.

[Question]
{question}

Instructions:
- Gather and list important facts or data related to the question.
- Since this is the information-gathering phase, do not provide any conclusions or in-depth reasoning yet.
- Include numbers or specific proper nouns as necessary.
- When presenting information, be sure to provide references or sources for each piece of information.
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
    logger.info(f"Response from GPT:\n{output_text}\n\n")

    return output_text

def step2_reasoning(question: str, knowledge_from_step1: str) -> str:
    """
    Step 2: Based on the knowledge from Step 1, perform logical reasoning.
    In this step, use the Pyramid Principle (Main Point, Sub Points, Supporting Data)
    and Chain of Thought (CoT) to organize your reasoning leading up to a conclusion.
    """
    prompt = f"""
You are a capable assistant. Please use the information below to reason about the question.

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}

Instructions:
1. Refer to the knowledge from Step 1 and structure it using Main Points and Sub Points in a hierarchical manner.
2. Based on that information, perform logical reasoning (Chain of Thought) up to the point just before you derive a final conclusion.
3. At this stage, do not provide the final conclusion. Instead, clarify what evidence supports your reasoning,
   how you compare or evaluate the data, and outline the Sub Points or Supporting Data under the Pyramid Principle.
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
    logger.info(f"Response from GPT:\n{output_text}\n\n")

    return output_text

def step3_final_answer(question: str, knowledge_from_step1: str, reasoning_from_step2: str) -> str:
    """
    Step 3: Integrate the results from Steps 1 and 2 to provide the final answer.
    """
    prompt = f"""
You are a capable assistant. Please use the following information to arrive at the final answer.

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}

[Reasoning from Step 2]
{reasoning_from_step2}

Instructions:
- Summarize the knowledge and reasoning in accordance with the Pyramid Principle (Main Point, Sub Points, Supporting Data) and provide the final conclusion.
- Present the final answer concisely, along with the key reasons that lead to it.
- Please use Arabic numerals (e.g., 1, 2, 3) when writing numbers, rather than spelling them out with alphabetic characters.
- Please write your final conclusion immediately after the '### Conclusion' section header.
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
    logger.info(f"Response from GPT:\n{output_text}\n\n\n\n")

    return output_text

def main(user_question: str) -> str:
    # ユーザーからの質問を受け取る
    # user_question = input("Please enter your question: ")
    logger.info(f"User's question: \n{user_question}\n\n")

    # Step 1: Knowledge retrieval
    knowledge = step1_knowledge_retrieval(user_question)
    # print("\n----- Step 1: Knowledge Retrieval -----")
    # print(knowledge)

    # Step 2: Reasoning and organization
    reasoning = step2_reasoning(user_question, knowledge)
    # print("\n----- Step 2: Reasoning and Organization -----")
    # print(reasoning)

    # Step 3: Final answer
    final_answer = step3_final_answer(user_question, knowledge, reasoning)
    # print("\n===== Final Answer =====")
    # print(final_answer)
    return final_answer

if __name__ == "__main__":
    main()
