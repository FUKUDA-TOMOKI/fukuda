import re
import json
from typing import List

def normalize_text(text: str) -> str:
    """
    前処理を行う:
      - 記号の正規化(コロン、カンマ、セミコロン、ハイフンなど)
      - 余計な空白の除去
      - 小文字化
    """
    # 1. 特定の記号をスペースに置換: （必要に応じて追加・調整）
    #   例：",", ":", ";", "-" を空白に変換
    text = re.sub(r"[\.,;:\-]", " ", text)

    # 2. 複数の空白を一つにまとめる
    text = re.sub(r"\s+", " ", text)

    # 3. 全て小文字に変換
    text = text.strip().lower()

    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    レーベンシュタイン距離（編集距離）を計算する。
    動的計画法で実装。
    """
    # Pythonic に書くなら、外部ライブラリ python-Levenshtein や rapidfuzz などがあるが、
    # ここでは純粋なDPで実装している。
    len_s1, len_s2 = len(s1), len(s2)

    # 距離を格納するための2次元リスト (サイズ (len_s1+1) x (len_s2+1))
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # 初期化
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
                dp[i - 1][j - 1] + cost  # 置換(文字が同じ場合はcost=0)
            )

    return dp[len_s1][len_s2]


def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    テキストを単語ごとに分割し、chunk_sizeずつのスライドウィンドウで区切ったフレーズのリストを返す。
    例：
      text = "the old man lives in a castle"
      chunk_size = 2
      -> ["the old", "old man", "man lives", "lives in", "in a", "a castle"]
    """
    words = text.split()
    chunks = []
    # スライドしながら chunk_size 語をまとめる
    for i in range(len(words) - chunk_size + 1):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def score_single_answer(correct_answer: str, user_answer: str) -> int:
    """
    単一型の正解に対して、ユーザーの回答を区切りながら最小のレーベンシュタイン距離を求める。
    - correct_answer は前処理済みの単一正解
    - user_answer は前処理済みの回答文
    """
    # 正解の単語数
    correct_words = correct_answer.split()
    chunk_size = len(correct_words)

    # 回答文を chunk_size 単語ずつスライドしながら区切る
    user_chunks = split_into_chunks(user_answer, chunk_size)

    if not user_chunks:
        # もしユーザー回答があまりに短く、chunk が得られない場合
        return levenshtein_distance(correct_answer, user_answer)

    # レーベンシュタイン距離を計算し、その最小値を返す
    min_dist = float('inf')
    for chunk in user_chunks:
        dist = levenshtein_distance(correct_answer, chunk)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def score_multiple_answers(correct_answers: List[str], user_answer: str) -> float:
    """
    複数型の正解それぞれに対して score_single_answer を計算し、
    その平均値をスコアとして返す。
    """
    if not correct_answers:
        return 0.0  # 正解が空リストならばスコア0など、状況に応じて決める

    total_score = 0
    for ans in correct_answers:
        dist = score_single_answer(ans, user_answer)
        total_score += dist

    return total_score / len(correct_answers)


def is_enumerated_answer(answer_data: dict) -> bool:
    """
    回答データ(answerフィールド)に複数要素があるかどうかを判定。
    複数要素なら列挙型(True)、1つなら単一型(False)。
    """
    answers = answer_data.get("answer", [])
    if answers is None:
        answers = []
    return len(answers) > 1


def evaluate_answer(correct_data: dict, user_answer: str) -> float:
    """
    JSON上の一問（正解データ）と、ユーザー回答文字列に対し、
    スコアを計算して返す。
    """
    # 1. 前処理
    user_answer_normalized = normalize_text(user_answer)

    # JSONから正解を取り出す
    answers = correct_data.get("answer", {})
    answer_list = answers.get("answer", [])

    # 答えのタイプが "enumerated"（複数型） かどうか
    # is_enumerated_answer() 関数で判定
    if is_enumerated_answer(correct_data.get("answer", {})):
        # 2-a. 複数型のケース
        # 仮に answer_list が文字列ならカンマ split でリストにするが、
        # データ形式が既に複数リストの場合はそのまま使う。
        # 下の例では answer_list は ["Paris", "London", "Tokyo"] のような形を想定。
        correct_answers = [normalize_text(ans) for ans in answer_list]
        score = score_multiple_answers(correct_answers, user_answer_normalized)
    else:
        # 2-b. 単一型のケース
        # answer_list が 1つの場合を想定。
        if not answer_list:
            return 0.0  # 正解が空の場合の例外処理

        correct_answer = normalize_text(str(answer_list[0]))
        score = score_single_answer(correct_answer, user_answer_normalized)

    return score


def main():
    # 例示用のダミー問題データ
    data_json = [
        {
            "id": "example_1",
            "question": "What is the phrase for an older male human?",
            "answer": {
                "answerType": "entity",
                "answer": ["old man"]  # 単一型
            }
        },
        {
            "id": "example_2",
            "question": "Name three famous cities in the world.",
            "answer": {
                "answerType": "entity",
                "answer": ["Paris", "London", "Tokyo"]  # 複数型
            }
        }
    ]

    # それぞれに対しユーザー回答を仮定してスコアを計算してみる
    user_answers = [
        "The old man lives in a castle.",
        "I like London, but Paris is also nice. Tokyo is far away."
    ]

    for idx, item in enumerate(data_json):
        question = item["question"]
        user_answer = user_answers[idx]

        print(f"Q: {question}")
        print(f"User answer: {user_answer}")

        score = evaluate_answer(item, user_answer)
        print(f"Score: {score}\n")


if __name__ == "__main__":
    main()
