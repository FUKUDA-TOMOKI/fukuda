import re
import json
from typing import List
from PoT_3 import main as pot_main
from CoT import main as cot_main

def normalize_text(text: str, remove_comma: bool = False) -> str:
    """
    前処理を行う:
      - 記号の正規化(コロン、セミコロン、ハイフンなど)
      - 余計な空白の除去
      - 小文字化
      - remove_comma=True の場合のみカンマを削除
    """
    if remove_comma:
        # カンマも含めて置換
        text = re.sub(r"[\.,;:\-]", " ", text)
    else:
        # カンマ以外を置換
        text = re.sub(r"[\.;:\-]", " ", text)

    # 複数の空白を1つにまとめる
    text = re.sub(r"\s+", " ", text)
    # 前後の空白を除去し、小文字化
    text = text.strip().lower()
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    レーベンシュタイン距離（編集距離）を計算する (動的計画法)。
    """
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 削除
                dp[i][j - 1] + 1,      # 挿入
                dp[i - 1][j - 1] + cost  # 置換 (同じ文字ならcost=0)
            )

    return dp[len_s1][len_s2]


def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    テキストを単語ごとに分割し、chunk_sizeごとにスライドさせながら区切る。
    """
    words = text.split()
    chunks = []
    for i in range(len(words) - chunk_size + 1):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def score_single_answer(correct_answer: str, user_answer: str) -> int:
    """
    単一型の正解文字列とユーザー回答を比較し、
    スライドさせた部分文字列との最小レーベンシュタイン距離を返す。
    """
    correct_words = correct_answer.split()
    chunk_size = len(correct_words)

    user_chunks = split_into_chunks(user_answer, chunk_size)
    if not user_chunks:
        # ユーザー回答が短すぎる場合など
        return levenshtein_distance(correct_answer, user_answer)

    min_dist = float('inf')
    for chunk in user_chunks:
        dist = levenshtein_distance(correct_answer, chunk)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def score_enumerated_answers(correct_answers: List[str], user_answer: str) -> float:
    """
    列挙型の正解（複数の正解要素）それぞれについて score_single_answer を計算し、
    その平均値を返す。
    """
    if not correct_answers:
        return 0.0

    total_score = 0
    for ans in correct_answers:
        dist = score_single_answer(ans, user_answer)
        total_score += dist
    return total_score / len(correct_answers)


def is_enumerated_answer(answer_mention: str) -> bool:
    """
    カンマが含まれていれば列挙型とみなす。
    """
    return ',' in answer_mention


def evaluate_answer(correct_mention: str, user_answer: str) -> float:
    """
    正解(mention文字列) と ユーザー回答 を受け取り、スコアを返す。
      - 列挙型(カンマが含まれる場合)はカンマで分割→各要素を単一型として平均スコア
      - 単一型はカンマを削除したうえで単一型スコアを算出
    """
    if is_enumerated_answer(correct_mention):
        # 列挙型: カンマで分割し、各要素を正規化(カンマは取り除かない設定は不要)
        # ただし、分割後は単一型としてスコア比較するので、個々の要素の正規化時には remove_comma=True でよい
        # → まず「列挙型を 'そのまま' 正規化してから split」するか、もしくは split してから個別に正規化するか
        #   ここでは先にカンマ含みのまま正規化→splitします
        mention_normalized = normalize_text(correct_mention, remove_comma=False)
        correct_items = [x.strip() for x in mention_normalized.split(',') if x.strip()]

        # ユーザー回答側は単一フレーズではないが、
        # 全体を「単一比較に使う形式」に正規化
        user_answer_normalized = normalize_text(user_answer, remove_comma=True)

        return score_enumerated_answers(correct_items, user_answer_normalized)
    else:
        # 単一型: カンマを削除して正規化
        correct_answer = normalize_text(correct_mention, remove_comma=True)
        user_answer_normalized = normalize_text(user_answer, remove_comma=True)
        return score_single_answer(correct_answer, user_answer_normalized)


def extract_questions_and_answers(data):
    """
    JSON の各要素から:
      - 質問文 (question)
      - 回答データ (answer_data)
      - 実際に用いる答え(mention)を取り出す
    """
    extracted_data = []
    for item in data:
        question = item.get("question", "Unknown Question")
        answer_data = item.get("answer", {})
        # 答えとして使うのは 'mention'
        answer_mention = answer_data.get("mention", "Unknown Answer")

        extracted_data.append({
            "question": question,
            "answer_data": answer_data,
            "answer": answer_mention
        })
    return extracted_data


def main():
    # ダミーデータ: 単一型/列挙型混在
    data_json = [
        {
            "id": "example_2",
            "question": "Name three biggest cities in the world.",
            "answer": {
                "answerType": "entity",
                # 列挙型 → カンマ区切り
                "mention": "New York, London, Tokyo"
            }
        }
    ]

    # JSONデータから question と正解(mention) を抽出
    extracted_data = extract_questions_and_answers(data_json)

    # 各質問に対して回答を生成し、スコアを計算
    for idx, entry in enumerate(extracted_data):
        question_text = entry["question"]
        correct_mention = entry["answer"]  # 実際の正解

        # ユーザー回答を生成（本来はユーザー入力取得など）
        pot_answer = pot_main(question_text)
        cot_answer = cot_main(question_text)

        print(f"Q{idx+1}: {question_text}")
        print(f"  Correct mention: {correct_mention}")
        print(f"  PoT answer: {pot_answer}")
        print(f"  CoT answer: {cot_answer}")

        pot_score = evaluate_answer(correct_mention, pot_answer)
        cot_score = evaluate_answer(correct_mention, cot_answer)
        print(f"  PoT score: {pot_score:.2f}")
        print(f"  CoT score: {cot_score:.2f}")


if __name__ == "__main__":
    main()
