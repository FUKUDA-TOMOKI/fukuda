import os
import logging
import openai
from dotenv import load_dotenv
from datetime import datetime
import json
import numpy as np
import faiss

# ログの設定（ファイル名には実行時の時刻を付与）
logging.basicConfig(
    filename=f'logs/pot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # ログファイル名
    filemode='a',                   
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# .envファイルの読み込み (API_KEY 等を取得)
load_dotenv()

# OpenAI APIキー
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

# GPT-4相当のモデル (ChatCompletion用)
MODEL_CHAT = "gpt-4o-mini"
# Embedding用の高性能モデル
MODEL_EMBEDDING = "text-embedding-3-large"

###############################################################################
# Embeddings APIを用いた文章ベクトル化
###############################################################################
def get_embedding(text: str, model: str = MODEL_EMBEDDING) -> np.ndarray:
    """
    OpenAIのEmbeddings APIを利用して、テキストをベクトル化する。
    返り値はfloat32のnumpy配列。
    """
    # 改行などを置き換え
    clean_text = text.replace("\n", " ")
    
    # encoding_format="float" を指定すると、embeddingが浮動小数点配列として返ってくる
    # Python用openaiライブラリでは返り値はdata[].embeddingにfloatのリストとして含まれる
    response = openai.Embedding.create(
        model=model,
        input=clean_text,
        encoding_format="float"
    )
    emb_list = response["data"][0]["embedding"]
    # numpy配列(float32)に変換
    emb_array = np.array(emb_list, dtype=np.float32)
    return emb_array


###############################################################################
# FAISS Indexの構築/検索 (title + 各sentenceを個別にEmbedding)
###############################################################################
def build_faiss_index(context: list):
    """
    コンテキスト（[title, sentences]のリスト）から、
    「title + 各sentence」をベクトル化し、FAISS Indexを構築する。

    Returns:
      index: FAISSインデックス
      meta: 各ベクトルに対応するメタ情報のリスト
            (title, sentence, context_idx, sentence_idx)などを格納
    """
    vectors = []
    meta = []

    for c_idx, (title, sentences) in enumerate(context):
        for s_idx, sentence in enumerate(sentences):
            combined_text = f"{title} {sentence}"
            emb = get_embedding(combined_text)
            vectors.append(emb)
            # メタ情報にはタイトル、文、オリジナルのインデックス等を保持
            meta.append((title, sentence, c_idx, s_idx))

    # ベクトルをNumPy配列にまとめる
    vectors_np = np.vstack(vectors)  # shape: (N, embedding_dim)
    dim = vectors_np.shape[1]

    # FAISSのIndex作成 (内積ベース)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    return index, meta

def retrieve_top_k(question: str, index, meta, k=3):
    """
    FAISSのIndexから、questionに最も類似度が高い上位k件を返す。
    結果は [ [title, [sentence1, ...]], ... ] の形式に合うようにまとめ直す。

    ただし、同じtitleが複数返ってくる場合があるため、
    タイトル単位で集約してから返却している。
    """
    # 質問をEmbedding化
    q_emb = get_embedding(question).reshape(1, -1)

    # 類似度検索
    distances, indices = index.search(q_emb, k)
    retrieved_info = []
    
    # 上位k件を集約
    # meta[i]は (title, sentence, context_idx, sentence_idx)
    for i in indices[0]:
        title, sentence, c_idx, s_idx = meta[i]
        retrieved_info.append((title, sentence))

    # タイトル単位で sentences をまとめる
    # 例: { "Some Title": ["Sentence1", "Sentence2", ...], ... }
    result_dict = {}
    for (title, sent) in retrieved_info:
        if title not in result_dict:
            result_dict[title] = []
        result_dict[title].append(sent)
    
    # 最終的には [ [title, [sentence1, sentence2, ...]], ... ]
    top_contexts = []
    for t, s_list in result_dict.items():
        top_contexts.append([t, s_list])

    return top_contexts


###############################################################################
# 多段推論用のサンプル関数
###############################################################################
def multi_step_context_retrieval(question: str, context: list, steps=2, top_k=3):
    """
    多段推論の例:
      1. コンテキストを細分化し、ベクトル化→FAISSでIndex作成
      2. 質問に対する上位k件を検索
      3. 検索結果から「中間的なサブクエリ」を作って再検索
    """
    # 1. FAISSインデックスの構築
    index, meta = build_faiss_index(context)

    # 2. 質問に対する上位k件を取得
    top_context_step1 = retrieve_top_k(question, index, meta, k=top_k)

    # サブクエリ例: 最初の検索結果の先頭文をquestionに付加して再検索
    if steps > 1 and len(top_context_step1) > 0:
        # top_context_step1[0] → [title, [sent1, sent2, ...]]
        first_title, first_sents = top_context_step1[0]
        if first_sents:
            sub_question = question + " " + first_sents[0]
        else:
            sub_question = question

        # 3. サブクエリに対して再度検索
        top_context_step2 = retrieve_top_k(sub_question, index, meta, k=top_k)
        return top_context_step1, top_context_step2
    else:
        return top_context_step1, []


###############################################################################
# ここから元のステップ関数 (GPTを使ったQAフロー)
###############################################################################
def step1_knowledge_retrieval(question: str, context: list) -> str:
    """
    Step 1: ユーザーの質問に関連する知識をMECE原則に基づいて収集する。
    """
    context_str = json.dumps(context, ensure_ascii=False, indent=2)
    prompt = f"""
You are a capable assistant. Your task is to collect key knowledge relevant to the following question using the MECE (Mutually Exclusive and Collectively Exhaustive) principle.
In this step, you are explicitly provided with information via the "context" section below.
Please output the information in the following bullet list format and limit your response to key points:
- **Category Name**:
  - **Item 1**: Explanation including specific numbers or proper nouns.
  - **Item 2**: Explanation.

Do not provide any conclusions or in-depth reasoning at this stage.

[Question]
{question}

[Context Provided]
{context_str}
"""

    logger.info("=== Step 1: Knowledge Retrieval ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content
    logger.info(f"Response from GPT (Step 1):\n{output_text}\n\n")
    return output_text

def step2_reasoning(question: str, knowledge_from_step1: str) -> str:
    """
    Step 2: Step 1の知識をもとに論理的推論を行う（ピラミッド原則 + Chain of Thought）。
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
    logger.info("=== Step 2: Reasoning and Organization ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content
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
    logger.info("=== Step 3: Final Answer ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content
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
    logger.info("=== Step 4: Final Answer Extraction ===")
    logger.info(f"Prompt to GPT:\n{prompt}\n\n")

    response = openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content
    logger.info(f"Response from GPT (Step 4):\n{output_text}\n\n")
    return output_text


###############################################################################
# メイン処理
###############################################################################
def main(user_question: str, context: list, top_k=3, multi_steps=1) -> str:
    logger.info(f"User's question:\n{user_question}\n\n")
    logger.info(f"Context (raw):\n{context}\n\n")

    # 多段推論を使うか、一段階の検索か
    if multi_steps > 1:
        top_context_step1, top_context_step2 = multi_step_context_retrieval(
            user_question, context, steps=multi_steps, top_k=top_k
        )
        # 両方の検索結果を合体させる（重複を排除するなどの処理も可）
        combined_context = top_context_step1 + top_context_step2
    else:
        index, meta = build_faiss_index(context)
        combined_context = retrieve_top_k(user_question, index, meta, k=top_k)

    logger.info(f"Filtered Context:\n{combined_context}\n\n")

    # Step 1: Knowledge retrieval
    knowledge = step1_knowledge_retrieval(user_question, combined_context)

    # Step 2: Reasoning and organization
    reasoning = step2_reasoning(user_question, knowledge)

    # Step 3: Final answer (detailed, including reasoning)
    full_final_answer = step3_final_answer(user_question, knowledge, reasoning)

    # Step 4: Extract only the final answer from the detailed answer
    final_answer = step4_extract_final_answer(user_question, full_final_answer)
    return final_answer


###############################################################################
# 実行例
###############################################################################
if __name__ == "__main__":
    # サンプルの質問
    sample_question = "Which Republican candidate ran for president in 2008 but did not win presidential primaries?"

    # サンプルのcontext
    sample_context = [
        [
          "2008 United States presidential election",
          [
            "The 2008 United States presidential election was the 56th quadrennial presidential election, held on Tuesday, November 4, 2008.",
            "Republican candidate John McCain ran for president in 2008 but did not win the presidential primaries."
          ]
        ],
        [
          "Sample Title",
          [
            "Sample text that might be irrelevant to the question.",
            "Another random sentence for demonstration."
          ]
        ]
    ]

    # 上位3件だけを利用して回答を生成
    result = main(sample_question, sample_context, top_k=3, multi_steps=1)
    print("=== Final Extracted Answer ===")
    print(result)
