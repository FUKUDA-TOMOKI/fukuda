import re
import json
import random
from typing import List
from scripts.PoT_5 import main as pot_main
from scripts.CoT import main as cot_main
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
    見つからなかった場合は、元の回答を返す。
    """
    marker = "### Conclusion"
    index = answer.find(marker)
    if index != -1:
        # marker の後ろの部分を返す (marker 自体は含まない)
        return answer[index + len(marker):].strip()
    else:
        return answer

def main():
    # 各answer typeごとに評価する対象数を指定
    # ※stringは件数が少ないので全問採用（8問）、その他は100問ずつ
    sample_sizes = {
        'entity': 100,
        'numerical': 100,
        'boolean': 100,
        'date': 100,
        'string': 8
    }
    answer_types = ['string', 'date', 'entity', 'boolean', 'numerical']

    # mintaka_test.jsonからデータをロード
    with open("data/mintaka_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # answer type ごとにデータをグループ分けする
    grouped_data = {atype: [] for atype in answer_types}
    for item in data:
        atype = item.get("answer", {}).get("answerType", None)
        if atype in grouped_data:
            grouped_data[atype].append(item)

    # 各answer typeごとにサンプルを抽出
    selected_data = []
    for atype in answer_types:
        items = grouped_data.get(atype, [])
        if atype == 'string':
            # stringは全件採用
            print(f"[{atype}] の問題数: {len(items)} (全問採用)")
            selected = items
        else:
            available = len(items)
            sample_size = sample_sizes.get(atype, 100)
            if available < sample_size:
                print(f"[{atype}] の問題数: {available} (希望数 {sample_size}問に満たないため全件採用)")
                selected = items
            else:
                print(f"[{atype}] の問題数: {available} → {sample_size}問をランダム抽出")
                selected = random.sample(items, sample_size)
        selected_data.extend(selected)

    # 抽出された全問題数の確認
    print(f"\n抽出された総問題数: {len(selected_data)}\n")

    # 全体の評価指標用
    pot_scores = []
    cot_scores = []

    # answerType ごとの評価指標を保持するための辞書を初期化
    pot_stats = {atype: {"count": 0, "total_score": 0.0} for atype in answer_types}
    cot_stats = {atype: {"count": 0, "total_score": 0.0} for atype in answer_types}

    # 各質問に対して回答を生成し、スコアを計算
    for idx, entry in enumerate(extract_questions_and_answers(selected_data)):
        question_text = entry["question"]
        correct_mention = entry["answer"]  # 実際の正解
        answer_type = entry["answer_type"]
        print(f"Q{idx+1} ({answer_type}): {question_text}")

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

        # 全体のスコア配列に追加
        pot_scores.append(pot_score)
        cot_scores.append(cot_score)

        # answerType ごとのスコア集計
        if answer_type in pot_stats:
            pot_stats[answer_type]["count"] += 1
            pot_stats[answer_type]["total_score"] += pot_score
        else:
            pot_stats.setdefault(answer_type, {"count": 1, "total_score": pot_score})
        if answer_type in cot_stats:
            cot_stats[answer_type]["count"] += 1
            cot_stats[answer_type]["total_score"] += cot_score
        else:
            cot_stats.setdefault(answer_type, {"count": 1, "total_score": cot_score})

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

    # answerType ごとの評価指標を計算して表示
    print("=== answerType ごとの評価指標 ===")
    print("PoT:")
    for atype in answer_types:
        count = pot_stats[atype]["count"]
        avg_score = pot_stats[atype]["total_score"] / count if count > 0 else 0
        print(f"  {atype}: 回数 = {count}, 平均スコア = {avg_score:.2f}")
    print("CoT:")
    for atype in answer_types:
        count = cot_stats[atype]["count"]
        avg_score = cot_stats[atype]["total_score"] / count if count > 0 else 0
        print(f"  {atype}: 回数 = {count}, 平均スコア = {avg_score:.2f}")

if __name__ == "__main__":
    main()
