import re
import json
import random
from typing import List
from PoT_3 import main as pot_main
from CoT import main as cot_main
from Levenshtein import distance as levenshtein_distance  # 外部モジュールを利用

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
        text = re.sub(r"[\.,;:*\"\-]", " ", text)
    else:
        # カンマ以外を置換
        text = re.sub(r"[\.;:*\"\-]", " ", text)

    # 複数の空白を1つにまとめる
    text = re.sub(r"\s+", " ", text)
    # 前後の空白を除去し、小文字化
    text = text.strip().lower()
    return text

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

    # min_distの最大値が1になるように正規化
    max_dist = len(correct_answer)
    return 1 - min(min_dist, max_dist) / max_dist

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
        # 列挙型: まず正規化後、カンマで分割
        mention_normalized = normalize_text(correct_mention, remove_comma=False)
        correct_items = [x.strip() for x in mention_normalized.split(',') if x.strip()]

        # ユーザー回答はカンマを除去して単一比較形式に正規化
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
        answer_type = answer_data.get("answerType", "Unknown Answer Type")

        extracted_data.append({
            "question": question,
            "answer_data": answer_data,
            "answer": answer_mention,
            "answer_type": answer_type
        })
    return extracted_data

def extract_final_answer(answer: str) -> str:
    """
    回答文から「### Conclusion」という文字列以降の部分のみを抽出する。
    見つからなかった場合は、その旨を print して元の回答を返す。
    """
    marker = "### Conclusion"
    index = answer.find(marker)
    if index != -1:
        # marker の後ろの部分を返す (marker 自体は含まない)
        return answer[index + len(marker):].strip()
    else:
        print("『### Conclusion』が見つかりませんでした。")
        return answer

def main():
    # ここで問題数を指定（例: 10問）
    num_questions = 50

    # mintaka_test.jsonからデータをロード
    with open("mintaka_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 指定された問題数分、ランダムに（重複なく）選択
    if num_questions > len(data):
        num_questions = len(data)
    selected_data = random.sample(data, num_questions)

    extracted_data = extract_questions_and_answers(selected_data)

    # PoT Score と CoT Score の配列を初期化
    pot_scores = []
    cot_scores = []

    # 各質問に対して回答を生成し、スコアを計算
    for idx, entry in enumerate(extracted_data):
        question_text = entry["question"]
        correct_mention = entry["answer"]  # 実際の正解
        answer_type = entry["answer_type"]
        print(f"Q{idx+1}: {question_text}")

        # ユーザー回答を生成（実際はユーザー入力などを利用）
        pot_answer = pot_main(question_text, answer_type)
        cot_answer = cot_main(question_text, answer_type)

        # 「### Conclusion」以降の部分のみを抽出
        pot_answer_final = extract_final_answer(pot_answer)
        cot_answer_final = extract_final_answer(cot_answer)

        print(f"  Correct mention: {correct_mention}")
        print(f"  PoT answer (final): {pot_answer_final}")
        print(f"  CoT answer (final): {cot_answer_final}")

        pot_score = evaluate_answer(correct_mention, pot_answer_final)
        cot_score = evaluate_answer(correct_mention, cot_answer_final)
        print(f"  PoT score: {pot_score:.2f}")
        print(f"  CoT score: {cot_score:.2f}")

        # 配列にスコアを追加
        pot_scores.append(pot_score)
        cot_scores.append(cot_score)

        # 現在までの平均スコアを計算して表示
        avg_pot = sum(pot_scores) / len(pot_scores)
        avg_cot = sum(cot_scores) / len(cot_scores)
        print(f"  現在までの平均 PoT score: {avg_pot:.2f}")
        print(f"  現在までの平均 CoT score: {avg_cot:.2f}\n")

if __name__ == "__main__":
    main()
