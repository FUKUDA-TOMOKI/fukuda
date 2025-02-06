import json

def output_first_hard_problem(problems):
    """
    問題リストから 'level' が 'hard' である最初の問題を出力する
    """
    for problem in problems:
        if problem.get('level') == 'easy':
            # 整形して出力するために JSON 形式で表示
            print(json.dumps(problem, indent=2, ensure_ascii=False))
            return  # 最初の一件のみ出力するのでループを終了

    print("Hard level の問題は見つかりませんでした。")

def main():
    # JSONファイルを読み込みます
    with open('data/hotpot_train_v1.1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 読み込んだデータから hard レベルの最初の問題を出力します
    output_first_hard_problem(data)

if __name__ == '__main__':
    main()
