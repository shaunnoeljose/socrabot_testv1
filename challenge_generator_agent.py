# challenge_generator_agent.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

@tool
def challenge_generator_tool(topic: str, difficulty_level: int) -> str:
    """
    Generates a fill-in-the-blanks Python programming challenge for the user.
    Use this when the user indicates they understand a topic and want to test their knowledge,
    or asks for a challenge/practice problem.
    Provide a Python code snippet with blanks to fill in, clearly marked (e.g., using '___').
    """
    challenge_difficulty_map = {
        0: "a very simple, single-line conceptual fill-in-the-blank",
        1: "a basic, 2-3 line code snippet challenge focusing on core syntax or a single concept",
        2: "an intermediate, 3-5 line code snippet challenge involving a few interconnected concepts or common patterns"
    }
    mapped_difficulty = challenge_difficulty_map.get(difficulty_level, "basic")

    challenge_prompt = (
        f"You are a Python programming challenge generator. Create {mapped_difficulty} "
        f"fill-in-the-blanks Python code snippet related to the topic: '{topic}'. "
        f"The challenge should be clear, concise, and directly relevant to the topic and difficulty level. "
        f"Ensure there is only ONE correct answer for each blank. "
        f"Mark blanks clearly using '___' (three underscores). "
        f"Provide a brief, clear instruction for the user on what to fill in. "
        f"Do NOT provide the answers or explanations after the challenge. "
        f"Example format:\n"
        f"Fill in the blanks to define a variable and print it:\n"
        f"```python\n"
        f"my_number = ___\n"
        f"print(my_number___)\n"
        f"```"
        f"\nNow, generate the challenge for the user on the topic '{topic}' at a {mapped_difficulty} level."
    )

    challenge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    try:
        response = challenge_llm.invoke(challenge_prompt)
        return "Here's a fill-in-the-blanks challenge for you based on our topic:\n\n" + response.content # type: ignore
    except Exception as e:
        print(f"Error during challenge generation: {e}")
        return "Could not generate a challenge at this moment."

