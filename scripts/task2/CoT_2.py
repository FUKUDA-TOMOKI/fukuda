import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import json
import numpy as np
import faiss

# PoT_2.py から必要な関数・変数をインポート
# （「PoT_2.py」が同じディレクトリにある想定）
from PoT_2 import (
    sbert_model,         # sentence-transformersのSentenceTransformerインスタンス
    build_faiss_index,   # FAISSインデックスを作る関数
    retrieve_top_k,      # FAISSインデックスからトップk件を検索する関数
    unify_contexts,      # 結果をタイトル単位でまとめる関数
    multi_step_context_retrieval  # 多段推論で検索結果を再取得する関数
)

# OpenAI クライアントの初期化
from openai import OpenAI

# ログの設定（ファイル名には実行時の時刻を付与）
logging.basicConfig(
    filename=f'logs/pot_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # ログファイル名
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# .envファイルの読み込み（API_KEY等を取得）
load_dotenv()
API_KEY = os.getenv("API_KEY")

# OpenAIクライアントを初期化
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def simple_gpt_answer(question: str, context: list) -> str:
    """
    context をテキストとして整形し、GPT へシンプルに投げる関数。
    ・Chain-of-Thought での4ステップは行わず、最終回答のみを返す。
    """
    # context情報を整形
    context_text = ""
    for title, info_list in context:
        context_text += f"{title}:\n"
        for info in info_list:
            context_text += f" - {info}\n"
        context_text += "\n"

    # Promptを作成
    prompt = (
        f"Context:\n{context_text}"
        f"Question: {question}\n"
        "Please analyze the above context and provide a concise final answer.\n"
        "Note: Do not show any hidden chain-of-thought process."
    )

    logger.info("=== Sending prompt to GPT (simple_gpt_answer) ===")
    logger.info(prompt)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    answer_text = response.choices[0].message.content.strip()
    logger.info("=== GPT Response (simple_gpt_answer) ===")
    logger.info(answer_text)

    return answer_text

def main(user_question: str, context: list, top_k=4, multi_steps=3) -> str:
    """
    user_question: ユーザーからの質問
    context:       [[title, [sentence1, ...]], [title2, [...]], ...]
    top_k:         1回の検索で返す件数
    multi_steps:   検索を何段階行うか(2以上で多段推論)
    
    1. 質問に対して FAISS で関連情報を検索
    2. 必要に応じて多段検索を実行
    3. 上位の文脈情報を GPT に投げて回答を得る
    """
    logger.info(f"User's question: {user_question}")
    logger.info(f"Context (raw): {json.dumps(context, ensure_ascii=False, indent=2)}")

    # 多段検索 or 単段検索
    if multi_steps > 1:
        # 多段検索で関連文脈を取得
        unified_context = multi_step_context_retrieval(
            user_question, context, steps=multi_steps, top_k=top_k
        )
    else:
        # 単段検索で上位k件を取得
        index, meta = build_faiss_index(context)
        top_k_context = retrieve_top_k(user_question, index, meta, k=top_k)
        unified_context = top_k_context

    logger.info(f"Filtered Context: {json.dumps(unified_context, ensure_ascii=False, indent=2)}")

    # シンプルな GPT 呼び出し
    final_answer = simple_gpt_answer(user_question, unified_context)
    return final_answer

if __name__ == "__main__":
    """
    デモ実行：
    sample_question と sample_context を元に main() を呼び出して
    結果の回答を表示する。
    """
    sample_question = "Who was born first out of Ash Lieb and Robert Frost?"
    sample_context = [
        ['Robert Frost House',
         [
             'The Robert Frost House is an historic house at 29-35 Brewster Street in Cambridge, Massachusetts.',
             'It was built in 1884.',
             'The house was home to poet Robert Frost for the last two decades of his life.'
         ]
        ],
        ['Birches (poem)',
         [
             '"Birches" is a poem by American poet Robert Frost (1874-1963).',
             'It was collected in Frost\'s third collection of poetry "Mountain Interval" (1916).'
         ]
        ],
        ['Ash Lieb',
         [
             'Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian.',
             'Born in Ballarat, Ash Lieb began exhibiting art at eight years of age.'
         ]
        ],
        ['Robert Frost',
         [
             'Robert Lee Frost (March 26, 1874 – January 29, 1963) was an American poet.',
             'He frequently wrote about rural life in New England.'
         ]
        ]
    ]

    answer = main(sample_question, sample_context, top_k=4, multi_steps=3)
    print("=== Final Answer ===")
    print(answer)
