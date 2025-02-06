import re
import json

def is_enumerated(answer_data: dict, question_text: str) -> bool:
    """
    回答が列挙型（複数項目）か単一型かを、回答リストの要素数のみで判定する関数。
    回答リストが空の場合は、その問題と回答データを表示する。

    :param answer_data: 回答のデータ。キー 'answer' にリストが入っていると想定。
    :param question_text: 質問文のテキスト。
    :return: 列挙型なら True、単一型なら False。
    """
    answers = answer_data.get("answer", [])
    if answers is None:
        answers = []
    
    return len(answers) > 1

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_questions_and_answers(data):
    """
    JSON の各要素から質問文と回答データ、または回答の一部（ここでは 'mention'）を抽出する。
    is_enumerated に渡すために、answer_data も合わせて保持する。
    """
    extracted_data = []
    for item in data:
        question = item.get("question", "Unknown Question")
        answer_data = item.get("answer", {})
        answer_mention = answer_data.get("mention", "Unknown Answer")
        extracted_data.append({
            "question": question,
            "answer_data": answer_data,
            "answer": answer_mention
        })
    return extracted_data

def main():
    file_path = "data/mintaka_test.json"  # JSONファイルのパスを指定
    data = load_json(file_path)
    extracted_data = extract_questions_and_answers(data)
    
    print(f"Total number of questions: {len(extracted_data)}")
    print("=== 判定結果 ===")
    for i, item in enumerate(extracted_data, 1):
        question = item["question"]
        answer = item["answer"]
        enum_result = is_enumerated(item["answer_data"], question)
        result_str = "Enumerated (列挙型)" if enum_result else "Single (単一型)"
        print(f"{i}. Question: {question}")
        print(f"   Answer: {answer}")
        print(f"   判定結果: {result_str}\n")

if __name__ == "__main__":
    main()
