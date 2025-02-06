import json
import random

def main():
    # JSONファイルを読み込みます
    with open('data/hotpot_train_v1.1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データ数が100件未満の場合は全件、100件以上の場合はランダムに100件選択
    sample_size = min(100, len(data))
    sample_data = random.sample(data, sample_size)
    
    # 選んだ各データのanswer属性を表示します
    for item in sample_data:
        print(item.get("answer"))

if __name__ == '__main__':
    main()
