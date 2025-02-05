import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_questions_and_answers(data):
    extracted_data = []
    for item in data:
        question = item.get("question", "Unknown Question")
        answer_data = item.get("answer", {})
        # 答えとして使うのは 'mention'
        answer_mention = answer_data.get("mention", "Unknown Answer")
        answer_type = answer_data.get("answerType", "Unknown Answer Type")
        # complexityTypeはquestionと同じ階層にある
        complexity = item.get("complexityType", "Unknown Complexity")
        extracted_data.append({
            "question": question,
            "answer_data": answer_data,
            "answer": answer_mention,
            "answer_type": answer_type,
            "complexity": complexity
        })
    return extracted_data

# answer_typeの一覧を取得する関数
def get_answer_types(data):
    answer_types = set()
    for item in data:
        answer_type = item.get("answer", {}).get("answerType", "Unknown Answer Type")
        answer_types.add(answer_type)
    return answer_types

# answer_typeのそれぞれの問題数をカウントする関数
def count_answer_types(data):
    answer_type_count = {}
    for item in data:
        answer_type = item.get("answer", {}).get("answerType", "Unknown Answer Type")
        if answer_type in answer_type_count:
            answer_type_count[answer_type] += 1
        else:
            answer_type_count[answer_type] = 1
    return answer_type_count

# complexityTypeの一覧を取得する関数
def get_complexity_types(data):
    complexity_types = set()
    for item in data:
        complexity = item.get("complexityType", "Unknown Complexity")
        complexity_types.add(complexity)
    return complexity_types

# complexityTypeのそれぞれの問題数をカウントする関数
def count_complexity_types(data):
    complexity_type_count = {}
    for item in data:
        complexity = item.get("complexityType", "Unknown Complexity")
        if complexity in complexity_type_count:
            complexity_type_count[complexity] += 1
        else:
            complexity_type_count[complexity] = 1
    return complexity_type_count

def main():
    file_path = "mintaka_test.json"  # JSONファイルのパスを指定
    data = load_json(file_path)
    
    # answerTypeの情報
    answer_type_all = get_answer_types(data)
    print(f"Answer types: {answer_type_all}")
    answer_type_count = count_answer_types(data)
    print(f"Answer type count: {answer_type_count}")
    
    # complexityTypeの情報
    complexity_type_all = get_complexity_types(data)
    print(f"Complexity types: {complexity_type_all}")
    complexity_type_count = count_complexity_types(data)
    print(f"Complexity type count: {complexity_type_count}")

if __name__ == "__main__":
    main()
