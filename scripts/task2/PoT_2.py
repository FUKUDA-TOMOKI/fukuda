import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import json
import numpy as np
import faiss

# sentence-transformers を用いたローカル埋め込み計算
from sentence_transformers import SentenceTransformer

# OpenAI クライアントの初期化
from openai import OpenAI

# ログの設定（ファイル名には実行時の時刻を付与）
logging.basicConfig(
    filename=f'logs/pot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # ログファイル名
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# .envファイルの読み込み（API_KEY等を取得）
load_dotenv()
API_KEY = os.getenv("API_KEY")

# OpenAIクライアントを初期化（参考コードに倣う）
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

###############################################################################
# sentence-transformersモデルのロード
###############################################################################
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
sbert_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

###############################################################################
# Embeddings（ローカルで計算）
###############################################################################
def get_embeddings(texts):
    """
    sentence-transformers を使ってテキスト配列を一括で埋め込み取得する。
    返り値は (N, embedding_dim) の numpy.ndarray (dtype: float32)。
    """
    embeddings = sbert_model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    return embeddings

###############################################################################
# FAISS Indexの構築
###############################################################################
def build_faiss_index(context):
    """
    context: [ [title, [sentence1, sentence2, ...]], ... ] の形式
    
    各「title + sentence」をまとめて埋め込み（バッチ処理）し、FAISS Indexを構築。
    Returns:
      index: FAISSインデックス
      meta:  各ベクトルに対応するメタ情報のリスト (title, sentence, context_idx, sentence_idx)
    """
    combined_texts = []
    meta = []
    
    for c_idx, (title, sentences) in enumerate(context):
        for s_idx, sentence in enumerate(sentences):
            combined_text = f"{title} {sentence}"
            combined_texts.append(combined_text)
            meta.append((title, sentence, c_idx, s_idx))
    
    vectors_np = get_embeddings(combined_texts)
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    return index, meta

###############################################################################
# FAISSからのトップk件検索
###############################################################################
def retrieve_top_k(question, index, meta, k):
    """
    質問に対して類似度が高い上位 k 件を FAISS から返す。
    結果は [ [title, [sentence1, ...]], ... ] の形式でまとめる。
    """
    q_emb = get_embeddings([question])
    distances, indices = index.search(q_emb, k)
    
    retrieved_info = []
    for i in indices[0]:
        title, sentence, c_idx, s_idx = meta[i]
        retrieved_info.append((title, sentence))
    
    result_dict = {}
    for (title, sent) in retrieved_info:
        if title not in result_dict:
            result_dict[title] = []
        result_dict[title].append(sent)
    
    top_contexts = []
    for t, s_list in result_dict.items():
        top_contexts.append([t, s_list])
    
    return top_contexts

###############################################################################
# コンテキストの統合（重複削除・タイトル単位でまとめる）
###############################################################################
def unify_contexts(*contexts):
    """
    contexts: 任意個数の [ [title, [sentence1, ...]], ... ] リストをまとめる
    タイトルごとに sentences を集約し、重複削除した上で最終的に
    [[title, [sentence1, ...]], [title2, [...]]] の形で返す。
    """
    unified_dict = {}
    for context in contexts:
        for item in context:
            title, sentences = item
            if title not in unified_dict:
                unified_dict[title] = set()
            for s in sentences:
                unified_dict[title].add(s)
    
    result = []
    for t, s_set in unified_dict.items():
        result.append([t, list(s_set)])
    
    return result

###############################################################################
# 多段推論の例
###############################################################################
def multi_step_context_retrieval(question, context, steps=3, top_k=4):
    """
    多段推論:
      1. コンテキストを細分化し、FAISSでIndex作成
      2. 質問に対する上位 k 件を検索 (top_context_step1)
      3. 2 の結果からサブクエリを作成し、各サブクエリで再検索
      4. 1回目と2回目の結果を統合して返す
    """
    index, meta = build_faiss_index(context)
    top_context_step1 = retrieve_top_k(question, index, meta, k=top_k)
    
    if steps > 1 and len(top_context_step1) > 0:
        top_contexts_step2_list = []
        for item in top_context_step1:
            title, sents = item
            if sents:
                sub_query = question + " " + sents[0]
            else:
                sub_query = question
            sub_context = retrieve_top_k(sub_query, index, meta, k=top_k)
            top_contexts_step2_list.append(sub_context)
        
        flattened_step2 = []
        for ctx in top_contexts_step2_list:
            flattened_step2.extend(ctx)
        
        unified_result = unify_contexts(top_context_step1, flattened_step2)
        return unified_result
    else:
        return top_context_step1

###############################################################################
# GPT 呼び出し（QAフロー）
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

    response = client.chat.completions.create(
        model=MODEL,
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

    response = client.chat.completions.create(
        model=MODEL,
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

    response = client.chat.completions.create(
        model=MODEL,
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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    output_text = response.choices[0].message.content
    logger.info(f"Response from GPT (Step 4):\n{output_text}\n\n")
    return output_text

###############################################################################
# メイン関数
###############################################################################
def main(user_question: str, context: list, top_k=4, multi_steps=3) -> str:
    """
    user_question: ユーザーからの質問
    context:       [[title, [sentence1, ...]], [title2, [...]], ...] 形式の文書
    top_k:         1回の検索で返す件数
    multi_steps:   検索を何段階行うか(2以上で多段推論)
    """
    logger.info(f"User's question:\n{user_question}\n\n")
    logger.info(f"Context (raw):\n{context}\n\n")

    if multi_steps > 1:
        unified_context = multi_step_context_retrieval(
            user_question, context, steps=multi_steps, top_k=top_k
        )
    else:
        index, meta = build_faiss_index(context)
        unified_context = retrieve_top_k(user_question, index, meta, k=top_k)
    
    logger.info(f"Filtered Context:\n{unified_context}\n\n")

    # Step 1: Knowledge retrieval
    knowledge = step1_knowledge_retrieval(user_question, unified_context)

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
    sample_question = "Who was born first out of Ash Lieb and Robert Frost?"
    sample_context = [
        ['Robert Frost House',
         ['The Robert Frost House is an historic house at 29-35 Brewster Street in Cambridge, Massachusetts.',
          'It consists of four wood-frame townhouses, 2-1/2 stories in height, arranged in mirror image styling.',
          'Each pair of units has a porch providing access to those units, supported by turned posts and with a low Stick style balustrade.',
          'The Queen Anne/Stick style frame house was built in 1884, and has gables decorated with a modest amount of Gothic-style bargeboard.',
          'The house was home to poet Robert Frost for the last two decades of his life.'
         ]],
        ['Birches (poem)',
         ['"Birches" is a poem by American poet Robert Frost(1874-1963).',
          'It was collected in Frost\'s third collection of poetry "Mountain Interval" that was published in 1916.',
          "Consisting of 59 lines, it is one of Robert Frost's most anthologized poems.",
          'The poem "Birches", along with other poems that deal with rural landscape and wildlife, shows Frost as a nature poet.'
         ]],
        ['Ash Lieb',
         ['Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.',
          'Born in Ballarat, Ash Lieb began exhibiting art at eight years of age, and at the age of fifteen, wrote his first novel, "The Secret Well".',
          'Throughout his career, Lieb has created a diverse range of artworks, books, short films, and comedic performances, often with philosophical or psychiatric undertones.'
         ]],
        ['Robert Frost',
         ['Robert Lee Frost (March 26, 1874 – January 29, 1963) was an American poet.',
          'He is highly regarded for his realistic depictions of rural life and his command of American colloquial speech.',
          'He frequently wrote about rural life in New England and used this setting to examine complex social and philosophical themes.'
         ]]
    ]

    answer = main(sample_question, sample_context, top_k=3, multi_steps=2)
    print("=== Final Extracted Answer ===")
    print(answer)
