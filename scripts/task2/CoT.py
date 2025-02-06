import os
from dotenv import load_dotenv
from openai import OpenAI

# .env ファイルの内容を読み込む
load_dotenv()

# 環境変数から API キーを取得
API_KEY = os.getenv("API_KEY")

# OpenAIクライアントを初期化
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def main(user_question: str, answer_type: str, context: list) -> str:
    # context引数に与えられた情報を整形してテキスト化する
    context_text = ""
    for title, info_list in context:
        context_text += f"{title}:\n"
        for info in info_list:
            context_text += f" - {info}\n"
        context_text += "\n"
    
    # PoTプロンプトを生成（context情報を明示的に含める）
    pot_prompt = (
        f"Context:\n{context_text}\n"
        + user_question +
        "\nLet's think step by step." +
        "\n- Please write only the final answer, not the reasoning process."
    )
    
    # answer_typeがbooleanの場合はYes/Noでの回答を求める
    if answer_type == "boolean":
        pot_prompt += "\n- Please answer with either 'Yes' or 'No' when the question can clearly be answered using one of those options."
    # answer_typeがnumericalの場合は全ての数字をアラビア数字で書くよう指示
    if answer_type == "numerical":
        pot_prompt += "\n- All numbers must be written using Arabic numerals."

    # ChatCompletion APIにリクエストを送信
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pot_prompt},
        ]
    )

    # 応答から最終的な回答を抽出して返す
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    # context情報の例
    context = [
        [
            "Verano de Escándalo (1998)",
            [
                "The 1998 Verano de Escándalo (Spanish for \"Summer of Scandal\") was the second annual \"Verano de Escandalo\" professional wrestling show promoted by AAA.",
                " The show took place on September 18, 1998, in Madero, Tamaulipas, Mexico.",
                " The main event featured steel cage match between the teams of Heavy Metal and Blue Demon Jr. and Kick Boxer and Abismo Negro.",
                " The stipulation of the main event was that if the team of Heavy Metal and Blue Demon Jr. lost referee Guicho Dominguez would be referee El Tirante's \"slave\" for a week.",
                " If Kick Boxer and Abismo Negro lost El Tirantes would be Guicho Dominguez's slave for a week."
            ]
        ],
        [
            "Triplemanía VII",
            [
                "Triplemanía VII was the seventh \"Triplemanía\" wrestling show promoted by AAA.",
                " The show took place on June 11, 1999, in Madero, Mexico.",
                " The Main event featured a Six-man \"Lucha Libre rules\" tag team match between the teams of Perro Aguayo, Octagón and El Cobarde II and El Texano, Perro Aguayo Jr. and Sangre Chicana.",
                " In the semi-main event Heavy Metal and El Felino defended the hair of their father, referee Pepe \"Tropi\" Casas while Kick Boxer and Thai Boxer defended the hair of referee El Tirantes.",
                " As a result, El Tirantes had his hair shaved off after the match."
            ]
        ],
        [
            "Protection racket",
            [
                "A protection racket is a scheme whereby a group provides protection to businesses or other groups through violence outside the sanction of the law.",
                " Through the credible threat of violence, the racketeers deter people from swindling, robbing, injuring, sabotaging or otherwise harming their clients.",
                " Protection rackets tend to appear in markets where the police and judiciary cannot be counted on to provide legal protection, either because of incompetence (as in weak or failed states) or illegality (black markets)."
            ]
        ],
        [
            "E. Gordon Gee",
            [
                "Elwood Gordon Gee (born February 2, 1944) is an American academic and is currently serving his second term as President of West Virginia University.",
                " He has served as the chief executive at several universities in the United States, previously serving at Ohio State University.",
                " Gee had been heading an Ohio State-based think tank following his retirement from the Ohio State presidency on July 1, 2013.",
                " He retired in response to a series of controversies relating to comments he made, the last of which involved anti-Catholic comments allegedly made in jest about the University of Notre Dame.",
                " His resignation thus ended his second term as the president; he had previously served as president of Ohio State from 1990 to 1997."
            ]
        ],
        [
            "Badr Hari",
            [
                "Badr Hari (Arabic: بدر هاري‎ ‎ ; born 8 December 1984) is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike's Gym in Oostzaan.",
                " He is a former K-1 Heavyweight champion (2007—2008), It's Showtime Heavyweight world champion (2009-2010) and \"K-1 World Grand Prix 2009\" finalist.",
                " Hari has been a prominent figure in the world of kickboxing and was once considered the best kickboxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring."
            ]
        ],
        [
            "Guerra de Titanes (1998)",
            [
                "\"Guerra de Titanes\" (1998) (\"War of the Titans\") was the second \"Guerra de Titanes\" professional wrestling show promoted by AAA.",
                " The show took place on December 13, 1998 in Chihuahua, Chihuahua, Mexico.",
                " The Main event featured a Steel Cage Match that highlighted two storyline feuds between Octagón and his \"Evil twin\" Pentagón and the feud between Heavy Metal and Kick Boxer as Octagón and Heavy Metal teamed together to take on Pentagón and Kick Boxer."
            ]
        ],
        [
            "Global Fighting Championship",
            [
                "Global Fighting Championship (also known as GFC) was a UAE-based kickboxing and mixed martial arts (MMA) event.",
                " Fighters from around world on the roster include Badr Hari, Peter Aerts, Peter Graham, Dewey Cooper, Zabit Samedov.",
                " It was considered as one of the biggest kickboxing and MMA promotion in Middle East.",
                "<ref name=\"Emirates 24/7\"> </ref>"
            ]
        ],
        [
            "Outrageous Betrayal",
            [
                "Outrageous Betrayal: The Dark Journey of Werner Erhard from est to Exile is a non-fiction book written by freelance journalist Steven Pressman and first published in 1993 by St. Martin's Press.",
                " The book gives an account of Werner H. Erhard's early life as Jack Rosenberg, his exploration of various forms of self-improvement techniques, and his foundation of Erhard Seminars Training \"est\" and later of Werner Erhard and Associates and of the Est successor course, \"The Forum\".",
                " Pressman details the rapid financial success Erhard had with these companies, as well as controversies relating to litigation involving former participants in his courses.",
                " The work concludes by going over the impact of a March 3, 1991 \"60 Minutes\" broadcast on CBS where members of Erhard's family made allegations against him, and Erhard's decision to leave the United States."
            ]
        ],
        [
            "Betting controversies in cricket",
            [
                "Cricket has had a number of controversies relating to players being involved with the betting aspects of the game.",
                " In particular, numerous players have been approached by bookmakers and bribed to throw matches, aspects of matches (e.g. the toss) or provide other information."
            ]
        ],
        [
            "Prosecution of gender-targeted crimes",
            [
                "Prosecution of gender-targeted crimes is the legal proceedings to prosecute crimes such as rape and domestic violence.",
                " The earliest documented prosecution of gender-based/targeted crimes is from 1474 when Sir Peter von Hagenbach was convicted for rapes committed by his troops.",
                " However, the trial was unsuccessful in indicting Sir von Hagenbach with the charge of rape because the war in which the rapes occurred was \"undeclared\" and thus the rapes were only considered illegal.",
                " Gender-targeted crimes continued to be prosecuted, but it was not until after World War II when an international criminal tribunal- the International Military Tribunal for the Far East (Tokyo Tribunal)- were officers charged for being responsible of the gender-targeted crimes (particularly rape) and other crimes against humanity.",
                " Despite the various rape charges, the Charter of the Tokyo Tribunal did not make references to rape, and rape was considered as subordinate to other war crimes.",
                " This is also the situation for other tribunals that followed, but with the establishments of the International Criminal Tribunal for the former Yugoslavia (ICTY) and the International Criminal Tribunal for Rwanda (ICTR), there was more attention to the prosecution of gender-targeted crimes with each of the statutes explicitly referring to rape and other forms of gender-targeted violence."
            ]
        ]
    ]
    user_question = "Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring."
    answer_type = "free"  # 例として自由形式の回答
    answer = main(user_question, answer_type, context)
    print(answer)
