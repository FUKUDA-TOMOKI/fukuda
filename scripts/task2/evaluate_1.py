import re
import json
import random
from typing import List
from PoT_1 import main as pot_main
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

def score_single_answer(correct_answer: str, user_answer: str) -> float:
    """
    単一型の正解文字列とユーザー回答を比較し、
    スライドさせた部分文字列との最小レーベンシュタイン距離を正規化した値を返す。
    """
    correct_words = correct_answer.split()
    chunk_size = len(correct_words)

    user_chunks = split_into_chunks(user_answer, chunk_size)
    max_dist = len(correct_answer)  # 正規化のための最大距離

    if not user_chunks:
        # ユーザー回答が短すぎる場合など
        dist = levenshtein_distance(correct_answer, user_answer)
        return 1 - min(dist, max_dist) / max_dist

    min_dist = float('inf')
    for chunk in user_chunks:
        dist = levenshtein_distance(correct_answer, chunk)
        if dist < min_dist:
            min_dist = dist

    return 1 - min(min_dist, max_dist) / max_dist

def evaluate_answer(correct_mention: str, user_answer: str) -> float:
    """
    正解(mention文字列) と ユーザー回答 を受け取り、スコアを返す。
    すべて単一型の回答として扱う。
    """
    correct_answer = normalize_text(correct_mention, remove_comma=True)
    user_answer_normalized = normalize_text(user_answer, remove_comma=True)
    return score_single_answer(correct_answer, user_answer_normalized)

def extract_questions_and_answers(data):
    """
    JSON の各要素から:
      - 質問文 (question)
      - 回答データ (answer_data)
      - 実際に用いる答え (answer)
      - レベル (level)
      - コンテキスト (context)
    を取り出す
    ※実際のデータ形式では、"answer" キーに直接文字列が入っている前提です。
    """
    extracted_data = []
    # data が辞書の場合はリストに包む
    if isinstance(data, dict):
        data = [data]
    for item in data:
        question = item.get("question", "Unknown Question")
        # "answer" キーが辞書の場合と文字列の場合に対応
        answer_val = item.get("answer", "Unknown Answer")
        if isinstance(answer_val, dict):
            answer_mention = answer_val.get("mention", "Unknown Answer")
        else:
            answer_mention = answer_val
        level = item.get("level", "Unknown Level")
        context = item.get("context", [])  # 追加：context を抽出

        extracted_data.append({
            "question": question,
            "answer_data": answer_val,
            "answer": answer_mention,
            "level": level,
            "context": context
        })
    return extracted_data

def main():
    # ファイルからデータをロード
    with open("data/mintaka_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # data がリスト形式でない場合はリストに変換
    if isinstance(data, dict):
        data = [data]

    # 今回は answer_type 属性は使用せず、全件採用する
    selected_data = data
    print(f"\n抽出された総問題数: {len(selected_data)}\n")

    # 全体の評価指標用
    pot_scores = []
    cot_scores = []

    # 単語数カウント用のリスト
    pot_word_counts = []
    cot_word_counts = []

    # level ごとの評価指標を保持するための辞書（動的にキーが決まる）
    pot_level_stats = {}
    cot_level_stats = {}

    # level ごとの単語数集計用辞書
    pot_words_by_level = {}
    cot_words_by_level = {}

    # 各質問に対して回答を生成し、スコアを計算
    for idx, entry in enumerate(extract_questions_and_answers(selected_data)):
        question_text = entry["question"]
        correct_answer = entry["answer"]  # 実際の正解
        level = entry["level"]
        context = entry["context"]  # コンテキストリスト
        print(f"Q{idx+1} ({level}): {question_text}")

        # ユーザー回答を生成（実際はユーザー入力などを利用）
        pot_answer_final = pot_main(question_text, context)
        cot_answer_final = cot_main(question_text, context)

        # 単語数をカウント（英語の回答であることを前提とする）
        pot_word_count = len(pot_answer_final.split())
        cot_word_count = len(cot_answer_final.split())
        pot_word_counts.append(pot_word_count)
        cot_word_counts.append(cot_word_count)

        # level ごとの単語数集計更新
        if level not in pot_words_by_level:
            pot_words_by_level[level] = {"count": 0, "total_words": 0}
        pot_words_by_level[level]["count"] += 1
        pot_words_by_level[level]["total_words"] += pot_word_count

        if level not in cot_words_by_level:
            cot_words_by_level[level] = {"count": 0, "total_words": 0}
        cot_words_by_level[level]["count"] += 1
        cot_words_by_level[level]["total_words"] += cot_word_count

        print(f"  Correct answer: {correct_answer}")
        print(f"  PoT answer (final): {pot_answer_final}")
        print(f"  CoT answer (final): {cot_answer_final}")
        print(f"  PoT word count: {pot_word_count}")
        print(f"  CoT word count: {cot_word_count}")

        pot_score = evaluate_answer(correct_answer, pot_answer_final)
        cot_score = evaluate_answer(correct_answer, cot_answer_final)
        print(f"  PoT score: {pot_score:.2f}")
        print(f"  CoT score: {cot_score:.2f}")

        # 全体のスコア配列に追加
        pot_scores.append(pot_score)
        cot_scores.append(cot_score)

        # level ごとのスコア集計
        if level not in pot_level_stats:
            pot_level_stats[level] = {"count": 0, "total_score": 0.0}
        pot_level_stats[level]["count"] += 1
        pot_level_stats[level]["total_score"] += pot_score

        if level not in cot_level_stats:
            cot_level_stats[level] = {"count": 0, "total_score": 0.0}
        cot_level_stats[level]["count"] += 1
        cot_level_stats[level]["total_score"] += cot_score

        # 現在までの平均スコアを計算して表示
        avg_pot = sum(pot_scores) / len(pot_scores)
        avg_cot = sum(cot_scores) / len(cot_scores)
        print(f"  現在までの平均 PoT score: {avg_pot:.2f}")
        print(f"  現在までの平均 CoT score: {avg_cot:.2f}\n")

    # 全体の評価指標表示
    print("=== 全体の評価指標 ===")
    overall_pot = sum(pot_scores) / len(pot_scores) if pot_scores else 0
    overall_cot = sum(cot_scores) / len(cot_scores) if cot_scores else 0
    print(f"PoT 全体平均スコア: {overall_pot:.2f}")
    print(f"CoT 全体平均スコア: {overall_cot:.2f}\n")

    # 単語数の平均値計算（全体）
    overall_pot_words = sum(pot_word_counts) / len(pot_word_counts) if pot_word_counts else 0
    overall_cot_words = sum(cot_word_counts) / len(cot_word_counts) if cot_word_counts else 0
    print("=== 単語数の平均値（全体） ===")
    print(f"PoT 回答の平均単語数: {overall_pot_words:.2f}")
    print(f"CoT 回答の平均単語数: {overall_cot_words:.2f}\n")

    # level ごとの評価指標を計算して表示
    print("=== level ごとの評価指標 ===")
    print("PoT:")
    for lvl, stats in pot_level_stats.items():
        count = stats["count"]
        avg_score = stats["total_score"] / count if count > 0 else 0
        word_stats = pot_words_by_level.get(lvl, {"count": 0, "total_words": 0})
        avg_words = word_stats["total_words"] / word_stats["count"] if word_stats["count"] > 0 else 0
        print(f"  {lvl}: 回数 = {count}, 平均スコア = {avg_score:.2f}, 平均単語数 = {avg_words:.2f}")
    print("CoT:")
    for lvl, stats in cot_level_stats.items():
        count = stats["count"]
        avg_score = stats["total_score"] / count if count > 0 else 0
        word_stats = cot_words_by_level.get(lvl, {"count": 0, "total_words": 0})
        avg_words = word_stats["total_words"] / word_stats["count"] if word_stats["count"] > 0 else 0
        print(f"  {lvl}: 回数 = {count}, 平均スコア = {avg_score:.2f}, 平均単語数 = {avg_words:.2f}")

if __name__ == "__main__":
    main()
