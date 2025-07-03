# code_explanation_agent.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

@tool
def code_explanation_tool(query: str) -> str:
    """
    Explains a Python concept, function, keyword, or code snippet.
    Use this when the user asks for a definition, explanation, or
    "what is the function of X?" for any Python-related term or code.
    Provide a clear, concise explanation suitable for a novice.
    """
    explanation_prompt = (
        f"You are a helpful Python programming explainer. Explain the following "
        f"Python concept, function, keyword, or code snippet concisely and clearly, "
        f"suitable for a novice programmer. Focus on its purpose and how it's used. "
        f"Do not ask questions. Provide a direct explanation.\n\nQuery: {query}"
    )
    
    # Using a separate LLM instance for explanation with a lower temperature for directness.
    explanation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    try:
        response = explanation_llm.invoke(explanation_prompt)
        return response.content
    except Exception as e:
        print(f"Error during code explanation: {e}")
        return "Could not explain this concept at the moment."

# Similar to code_analysis_agent, this can be a standalone tool function.
