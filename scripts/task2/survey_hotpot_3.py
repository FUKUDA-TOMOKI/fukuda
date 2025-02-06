import json
import random

def is_valid_answer(answer: str) -> bool:
    """
    answerにカンマが含まれており、
    かつすべてのカンマの直前の文字が数字でない場合にTrueを返す。
    """
    if "," not in answer:
        return False
    # answer内のすべてのカンマについて、直前の文字が数字でなければTrue
    for idx, char in enumerate(answer):
        if char == ",":
            # idxが0の場合は前に文字がないのでスキップ（今回は数字判定対象外）
            if idx > 0 and answer[idx - 1].isdigit():
                return False
    return True

def main():
    # JSONファイルを読み込みます
    with open('data/hotpot_train_v1.1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 全件数を表示
    total_count = len(data)
    print(f"全データ件数: {total_count}")
    
    # answerにカンマが含まれ、かつカンマの前の文字が数字でないデータをフィルタリング
    comma_answer_data = [item for item in data if is_valid_answer(item.get("answer", ""))]
    comma_count = len(comma_answer_data)
    print(f"answerにカンマを含む問題（カンマの前の文字が数字でない）の件数: {comma_count}")
    
    # 該当データが10件以上あればランダムに10件、少なければ全件を選択
    sample_size = min(10, comma_count)
    sample_data = random.sample(comma_answer_data, sample_size) if sample_size > 0 else []
    
    print("\nランダムに選んだ10件（context以外の属性）のデータ:")
    for item in sample_data:
        # context属性を除くデータを辞書内包表記で作成
        filtered_item = {k: v for k, v in item.items() if k != "context"}
        print(json.dumps(filtered_item, ensure_ascii=False, indent=2))
    
if __name__ == '__main__':
    main()
