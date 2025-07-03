# mcq_agent.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import re

@tool
def mcq_generator_tool(topic: str, difficulty_level: int) -> str:
    """
    Generates a multiple-choice question (MCQ) for a given Python topic and difficulty level.
    The output will include the question, 3-4 options (A, B, C, D), and the single correct answer.
    Use this when the user is struggling and needs a different way to test understanding.
    The response format will be:
    Question: [Question text]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Correct Answer: [A/B/C/D]
    """
    mcq_difficulty_map = {
        0: "a very simple, foundational multiple-choice question",
        1: "a basic multiple-choice question testing understanding of a core concept",
        2: "an intermediate multiple-choice question requiring a bit more thought or application"
    }
    mapped_difficulty = mcq_difficulty_map.get(difficulty_level, "basic")

    mcq_prompt = (
        f"You are a Python programming MCQ generator. Create {mapped_difficulty} "
        f"multiple-choice question related to the topic: '{topic}'. "
        f"The question should be clear, concise, and have only ONE correct answer. "
        f"Provide 3-4 distinct options labeled A, B, C, D. "
        f"Crucially, include the correct answer on a separate line at the very end in the format 'Correct Answer: [A/B/C/D]'. "
        f"Do NOT include any introductory or concluding remarks beyond the question and options."
        f"\n\nExample Output Format:\n"
        f"Question: What is Python primarily known for?\n"
        f"A. Mobile App Development\n"
        f"B. Web Scraping\n"
        f"C. Data Analysis\n"
        f"D. All of the above\n"
        f"Correct Answer: D"
        f"\n\nNow, generate the MCQ for the user on the topic '{topic}' at a {mapped_difficulty} level."
    )

    mcq_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    try:
        response = mcq_llm.invoke(mcq_prompt)
        return response.content
    except Exception as e:
        print(f"Error during MCQ generation: {e}")
        return "Could not generate a multiple-choice question at this moment."

# Function to parse the MCQ response and extract question, options, and correct answer
def parse_mcq_response(response_text: str):
    """Parses the raw LLM response for an MCQ into a structured dictionary."""
    mcq = {
        "question": "",
        "options": {},
        "correct_answer": ""
    }
    lines = response_text.strip().split('\n')
    
    if not lines:
        return None

    # Extract question
    if lines[0].startswith("Question:"):
        mcq["question"] = lines[0][len("Question:"):].strip()
        lines = lines[1:] # Remove question line

    # Extract options
    for line in lines:
        if re.match(r"^[A-D]\.", line):
            option_label = line[0]
            option_text = line[2:].strip()
            mcq["options"][option_label] = option_text
        elif line.startswith("Correct Answer:"):
            mcq["correct_answer"] = line[len("Correct Answer:"):].strip()
    
    if not mcq["question"] or not mcq["options"] or not mcq["correct_answer"]:
        return None # Parsing failed or incomplete MCQ

    return mcq