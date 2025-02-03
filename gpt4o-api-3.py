import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Retrieve API key from environment variables
API_KEY = os.getenv("API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def step1_knowledge_retrieval(question: str) -> str:
    """
    Step 1: Collect knowledge related to the question according to the MECE principle.
    """
    prompt = f"""
You are a capable assistant. First, collect knowledge relevant to the user's question.
Use the MECE (Mutually Exclusive and Collectively Exhaustive) principle to categorize the information,
and list the main points.

[Question]
{question}

Instructions:
- Gather and list important facts or data related to the question.
- Since this is the information-gathering phase, do not provide any conclusions or in-depth reasoning yet.
- Include numbers or specific proper nouns as necessary.
- When presenting information, be sure to provide references or sources for each piece of information.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def step2_reasoning(question: str, knowledge_from_step1: str) -> str:
    """
    Step 2: Based on the knowledge from Step 1, perform logical reasoning.
    In this step, use the Pyramid Principle (Main Point, Sub Points, Supporting Data)
    and Chain of Thought (CoT) to organize your reasoning leading up to a conclusion.
    """
    prompt = f"""
You are a capable assistant. Please use the information below to reason about the question.

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}

Instructions:
1. Refer to the knowledge from Step 1 and structure it using Main Points and Sub Points in a hierarchical manner.
2. Based on that information, perform logical reasoning (Chain of Thought) up to the point just before you derive a final conclusion.
3. At this stage, do not provide the final conclusion. Instead, clarify what evidence supports your reasoning,
   how you compare or evaluate the data, and outline the Sub Points or Supporting Data under the Pyramid Principle.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def step3_final_answer(question: str, knowledge_from_step1: str, reasoning_from_step2: str) -> str:
    """
    Step 3: Integrate the results from Steps 1 and 2 to provide the final answer.
    """
    prompt = f"""
You are a capable assistant. Please use the following information to arrive at the final answer.

[Question]
{question}

[Knowledge from Step 1]
{knowledge_from_step1}

[Reasoning from Step 2]
{reasoning_from_step2}

Instructions:
- Summarize the knowledge and reasoning in accordance with the Pyramid Principle (Main Point, Sub Points, Supporting Data) and provide the final conclusion.
- Present the final answer concisely, along with the key reasons that lead to it.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def main():
    # Get the user's question
    user_question = input("Please enter your question: ")

    # Step 1: Knowledge retrieval
    knowledge = step1_knowledge_retrieval(user_question)
    print("\n----- Step 1: Knowledge Retrieval -----")
    print(knowledge)

    # Step 2: Reasoning and organization
    reasoning = step2_reasoning(user_question, knowledge)
    print("\n----- Step 2: Reasoning and Organization -----")
    print(reasoning)

    # Step 3: Final answer
    final_answer = step3_final_answer(user_question, knowledge, reasoning)
    print("\n===== Final Answer =====")
    print(final_answer)

if __name__ == "__main__":
    main()
