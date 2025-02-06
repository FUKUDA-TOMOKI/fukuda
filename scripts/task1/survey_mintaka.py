import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_questions_and_answers(data):
    extracted_data = []
    for item in data:
        question = item.get("question", "Unknown Question")
        answer = item.get("answer", {}).get("mention", "Unknown Answer")
        extracted_data.append({"question": question, "answer": answer})
    return extracted_data

def find_top_longest_answers(extracted_data, top_n=5):
    sorted_data = sorted(extracted_data, key=lambda x: len(x["answer"].split()), reverse=True)
    return sorted_data[:top_n]

def main():
    file_path = "data/mintaka_test.json"  # JSONファイルのパスを指定
    data = load_json(file_path)
    extracted_data = extract_questions_and_answers(data)
    top_longest_answers = find_top_longest_answers(extracted_data, top_n=50)
    
    print(f"Total number of questions: {len(extracted_data)}")
    print("Top 50 longest answers:")
    for i, item in enumerate(top_longest_answers, 1):
        print(f"{i}. Question: {item['question']}")
        print(f"   Answer: {item['answer']}")
        print(f"   Word count: {len(item['answer'].split())}\n")

if __name__ == "__main__":
    main()
