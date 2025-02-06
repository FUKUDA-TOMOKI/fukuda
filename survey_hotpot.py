import json

def extract_sentence(data, specification):
    """
    data: 単一のデータエントリ（辞書）  
    specification: [title, sentence_index] の形式のリスト（例: ['University of Oklahoma', 0]）
    
    data の context 内から、指定された title を持つ段落を探し、
    指定された sentence_index の文章を返します。見つからなければ None を返します。
    """
    target_title, sentence_index = specification
    for paragraph in data.get('context', []):
        if paragraph[0] == target_title:
            sentences = paragraph[1]
            if 0 <= sentence_index < len(sentences):
                return sentences[sentence_index]
    return None

def main():
    # JSONファイルを読み込みます
    with open('data/hotpot_train_v1.1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 例として、最初のデータエントリを使用
    first_entry = data[900]
    
    # 任意の指定（例: ['University of Oklahoma', 0]）
    specification = ['Sam Bradford', 1]
    
    sentence = extract_sentence(first_entry, specification)
    
    if sentence:
        print("抽出された文章:")
        print(sentence)
    else:
        print("指定された文章が見つかりませんでした。")

if __name__ == '__main__':
    main()
