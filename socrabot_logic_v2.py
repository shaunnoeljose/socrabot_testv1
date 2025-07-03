# socratic_bot_logic.py
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

class SocraticBot:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Initializes the SocraticBot with a Gemini LLM and sets up LangChain agents.

        Args:
            api_key (str): The Google API key for Gemini.
            model_name (str): The name of the Gemini model to use.
            temperature (float): The sampling temperature for the LLM.
        """
        # Explicitly configure the generativeai library with the API key
        genai.configure(api_key=api_key) # type: ignore

        # Initialize the base LLM for the Socratic persona and tool calling
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.chat_history = [] # Stores HumanMessage and AIMessage objects
        self.difficulty = 0    # Initial difficulty level (0: Novice)
        self.current_topic = "variables in Python" # Track the current active learning topic

        # --- Define Tools (Simulated Agents) ---
        # The Code Analysis Tool
        @tool
        def code_analysis_tool(code_snippet: str) -> str:
            """
            Analyze a Python code snippet provided by the user.
            This tool provides concise, high-level feedback on potential issues,
            errors, or areas for improvement without fixing the code.
            Use this when the user explicitly provides Python code for review.
            """
            return self._perform_code_analysis(code_snippet)
        
        # The Code Explanation Tool (New Agent)
        @tool
        def code_explanation_tool(query: str) -> str:
            """
            Explains a Python concept, function, keyword, or code snippet.
            Use this when the user asks for a definition, explanation, or
            "what is the function of X?" for any Python-related term or code.
            Provide a clear, concise explanation suitable for a novice.
            """
            return self._perform_code_explanation(query)
        
        self.tools = [code_analysis_tool, code_explanation_tool]

        # --- Set up the LangChain Agent for Orchestration ---
        # The system prompt for the agent. This is crucial for controlling its behavior.
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_socratic_system_instruction()),
            MessagesPlaceholder(variable_name="chat_history"), # To maintain conversation context
            MessagesPlaceholder(variable_name="agent_scratchpad") # For agent internal thoughts/actions
        ])
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)
        
        # Create the AgentExecutor to run the agent
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False) # Set verbose to True for detailed agent logs

    def _get_socratic_system_instruction(self, hint_requested: bool = False) -> str:
        """
        Generates the system instruction for the LLM to guide its Socratic behavior
        and tool usage.
        """
        difficulty_hint = ""
        if self.difficulty == 0:
            difficulty_hint = "Keep your questions simple, foundational, and guide the user gently. Break down complex ideas into smaller, manageable parts."
        elif self.difficulty == 1:
            difficulty_hint = "Ask slightly more challenging questions, prompting deeper thought but still offering clear pathways. Use follow-up questions to probe understanding and connect concepts."
        elif self.difficulty == 2:
            difficulty_hint = "Pose challenging, open-ended questions that require more critical thinking and problem-solving. Encourage independent research or experimentation. Focus on nuanced understanding."
        else:
            difficulty_hint = "Be a helpful Socratic tutor."

        hint_instruction = ""
        if hint_requested:
            hint_instruction = "The user has explicitly asked for a hint. Provide a subtle clue or a guiding question that helps them move forward without giving away the direct answer. Ensure it's concise."

        # Instructions for tool usage
        tool_usage_instruction = (
            "If the user provides Python code or asks for code review, you MUST use the `code_analysis_tool` "
            "to get feedback on the code first. Then, use that feedback to formulate a Socratic question related to their code or the issues found."
            "If the user asks for an explanation of a Python concept, function (e.g., 'what is sort()?', 'explain lists'), "
            "or a code snippet explanation, you MUST use the `code_explanation_tool` to provide a concise explanation. "
            "After providing the explanation via the tool, return to Socratic questioning to deepen their understanding of that concept or function."
            "Otherwise, continue with Socratic questioning relevant to the current topic."
        )

        return (
            f"You are a Socratic Python programming tutor for novice learners. "
            f"Your primary goal is to guide the user to discover solutions and understand concepts through questioning, "
            f"aiming for efficient learning and quick grasp of concepts. "
            f"Do not provide direct answers or code solutions directly. "
            f"Always ask a follow-up question unless the user explicitly requests to exit. "
            f"The current topic is: {self.current_topic}. "
            f"{difficulty_hint} {hint_instruction} {tool_usage_instruction}"
        )

    # --- Simulated Agent: Code Analysis Agent Logic ---
    def _perform_code_analysis(self, code_snippet: str) -> str:
        """
        Internal method that simulates the Code Analysis Agent's function.
        Uses a separate LLM call for direct analysis feedback.
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
            return response.content # type: ignore
        except Exception as e:
            print(f"Error during code analysis: {e}")
            return "Could not perform code analysis at this moment."

    # --- Simulated Agent: Code Explanation Agent Logic ---
    def _perform_code_explanation(self, query: str) -> str:
        """
        Internal method that simulates the Code Explanation Agent's function.
        Provides clear, concise explanations of Python concepts or code.
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
            return response.content # type: ignore
        except Exception as e:
            print(f"Error during code explanation: {e}")
            return "Could not explain this concept at the moment."

    def send_message_to_llm(self, user_message: str, hint_requested: bool = False) -> str:
        """
        Sends the user's message to the LangChain AgentExecutor and returns the bot's response.
        The AgentExecutor will decide whether to use tools or respond directly.
        """
        # Recreate the agent/executor to ensure the system prompt reflects the latest state (e.g., hint_requested)
        self.agent = create_tool_calling_agent(self.llm, self.tools, ChatPromptTemplate.from_messages([
            ("system", self._get_socratic_system_instruction(hint_requested=hint_requested)),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]))
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False) # Set verbose to True for detailed agent logs

        try:
            result = self.agent_executor.invoke({
                "chat_history": self.chat_history,
                "input": user_message
            })
            bot_response_text = result["output"]
            return bot_response_text
        except Exception as e:
            print(f"Error communicating with AgentExecutor: {e}")
            return "I'm having trouble understanding right now. Can you please rephrase or try again?"

    def add_message_to_history(self, message):
        """Adds a message (HumanMessage or AIMessage) to the chat history."""
        self.chat_history.append(message)

    def get_difficulty(self) -> int:
        """Returns the current difficulty level."""
        return self.difficulty

    def set_difficulty(self, level: int):
        """Sets the difficulty level, ensuring it's within bounds."""
        self.difficulty = max(0, min(level, 2)) # Ensure difficulty stays between 0 and 2

    def adjust_difficulty_based_on_response(self, bot_response_text: str, hint_requested: bool):
        """
        Adjusts the difficulty level based on the bot's response length.
        This is a heuristic and can be replaced with more sophisticated logic.
        """
        if not hint_requested: # Only adjust difficulty based on regular responses, not hints
            if len(bot_response_text) < 100: # Shorter response might imply user is on track, increase difficulty
                self.set_difficulty(self.difficulty + 1)
            elif len(bot_response_text) > 200: # Longer response might imply user needs more guidance, decrease difficulty
                self.set_difficulty(self.difficulty - 1)

    def update_current_topic(self, topic: str):
        """Updates the current topic the bot is focusing on."""
        self.current_topic = topic
