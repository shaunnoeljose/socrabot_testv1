# code_analysis_agent.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

@tool
def code_analysis_tool(code_snippet: str) -> str:
    """
    Analyze a Python code snippet provided by the user.
    This tool provides concise, high-level feedback on potential issues,
    errors, or areas for improvement without fixing the code.
    Use this when the user explicitly provides Python code for review.
    """
    analysis_prompt = (
        f"You are a Python code analysis agent. Review the following Python code snippet "
        f"and provide concise, high-level feedback on potential issues, errors, or areas for improvement. "
        f"Do NOT fix the code or provide corrected code. Focus on pointing out where the user should look or what they should consider. "
        f"If the code looks reasonable for a beginner, provide encouraging feedback. "
        f"Code:\n```python\n{code_snippet}\n```"
    )
    
    # Using a separate LLM instance for analysis with a lower temperature for directness.
    analysis_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    try:
        response = analysis_llm.invoke(analysis_prompt)
        return response.content
    except Exception as e:
        print(f"Error during code analysis: {e}")
        return "Could not perform code analysis at this moment."

# You might define a class here if the agent needs internal state beyond what tools provide.
# For now, a standalone tool function is sufficient for this simple task.
