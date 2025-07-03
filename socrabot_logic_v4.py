import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Import the individual tool functions
from code_analysis_agent import code_analysis_tool
from code_explaination_agent import code_explanation_tool
from challenge_generator_agent import challenge_generator_tool
from mcq_agent import mcq_generator_tool, parse_mcq_response # Import new MCQ tool and parser


class SocraticBot:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Initializes the SocraticBot with a Gemini LLM and sets up LangChain agents.

        Args:
            api_key (str): The Google API key for Gemini.
            model_name (str): The name of the Gemini model to use.
            temperature (float): The sampling temperature for the LLM.
        """
        # Configuring the API key
        genai.configure(api_key=api_key) # type: ignore

        # Initialize the base LLM
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.chat_history = [] # Stores HumanMessage and AIMessage objects
        self.difficulty = 0    # Initial difficulty level (0: Novice)
        self.current_topic = "variables in Python" # Track the current active learning topic

        # --- Define Tools (Simulated Agents) ---
        self.tools = [code_analysis_tool, code_explanation_tool, challenge_generator_tool, mcq_generator_tool] # Add MCQ tool

        # --- Set up the LangChain Agent for Orchestration ---
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_socratic_system_instruction()),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False) # Set verbose to True for detailed agent logs

    def _get_socratic_system_instruction(self, hint_requested: bool = False) -> str:
        """
        Generates the system instruction for the LLM to guide its Socratic behavior
        and tool usage, including adaptation for user struggle with MCQs.
        """
        difficulty_hint = ""
        if self.difficulty == 0:
            difficulty_hint = "Keep your questions simple, foundational, and guide the user gently. " \
            "Break down complex ideas into smaller, manageable parts."
        elif self.difficulty == 1:
            difficulty_hint = "Ask slightly more challenging questions, prompting deeper thought but still offering clear pathways. " \
            "Use follow-up questions to probe understanding and connect concepts."
        elif self.difficulty == 2:
            difficulty_hint = "Pose challenging, open-ended questions that require more critical thinking and problem-solving. " \
            "Encourage independent research or experimentation. Focus on nuanced understanding."
        else:
            difficulty_hint = "Be a helpful Socratic tutor."

        hint_instruction = ""
        if hint_requested:
            hint_instruction = "The user has explicitly asked for a hint. Provide a subtle clue or a guiding question that helps " \
            "them move forward without giving away the direct answer. Ensure it's concise."

        # Updated instruction for handling struggle with MCQ option:
        struggle_instruction = (
            "If the user's response indicates confusion, frustration, or incorrect understanding "
            "after a challenge or a complex Socratic question, you have two options to help them: "
            "1. Immediately rephrase your Socratic question to be simpler and more fundamental. Break down the concept into smaller steps. "
            "2. Alternatively, offer a multiple-choice question (MCQ) related to the current topic and their struggle level.** "
            "To do this, you MUST use the `mcq_generator_tool` with the current `topic` and `difficulty`."
            "After the user answers the MCQ (the answer will be provided in the next turn), you will receive their response. "
            "If they answer correctly, provide positive reinforcement and then ask a slightly more difficult Socratic question related to the topic. "
            "If they answer incorrectly, provide gentle corrective feedback and then either offer to re-explain the concept with a simpler Socratic question or offer another MCQ. "
            "The goal is to help them gain confidence by succeeding at an easier step or by choosing correctly from options before returning to harder problems. "
            "Do NOT just repeat the same question. Adapt your difficulty in real-time."
        )

        # New instruction for progression after correct answers:
        progression_instruction = (
            "If the user consistently answers your Socratic questions correctly or successfully completes a challenge, "
            "recognize their understanding. Instead of repeating similar questions, smoothly transition to a related, "
            "slightly more advanced sub-concept within the current `current_topic`, or propose a new, slightly harder `current_topic`. "
            "Vary your Socratic questioning style: ask for implications, real-world examples, comparisons, or how they would apply the concept. "
            "Do NOT get stuck asking the same type of question about a concept they've clearly grasped."
        )

        # Instructions for tool usage
        tool_usage_instruction = (
            "If the user provides Python code or asks for code review, you MUST use the `code_analysis_tool` "
            "to get feedback on the code first. Then, use that feedback to formulate a Socratic question related to their code or the issues found."
            "If the user asks for an explanation of a Python concept, function (e.g., 'what is sort()?', 'explain lists'), "
            "or a code snippet explanation, you MUST use the `code_explanation_tool` to provide a concise explanation. "
            "After providing the explanation via the tool, return to Socratic questioning to deepen their understanding of that concept or function."
            "If the user indicates they understand the current topic (e.g., 'I understand', 'I got it', 'ready for challenge', 'give me a challenge', 'test me'), "
            "you MUST use the `challenge_generator_tool` to provide a fill-in-the-blanks challenge related to the `current_topic` and `difficulty`."
            "Otherwise, continue with Socratic questioning relevant to the current topic."
        )

        return (
            f"You are a Socratic Python programming tutor for novice learners. "
            f"Your primary goal is to guide the user to discover solutions and understand concepts through questioning, "
            f"aiming for efficient learning and quick grasp of concepts. "
            f"Do not provide direct answers or code solutions directly. "
            f"Always ask a follow-up question unless the user explicitly requests to exit. "
            f"The current topic is: {self.current_topic}. "
            f"{difficulty_hint} {hint_instruction} {struggle_instruction} {progression_instruction} {tool_usage_instruction}"
        )

    def send_message_to_llm(self, user_message: str, hint_requested: bool = False) -> str:
        """
        Sends the user's message to the LangChain AgentExecutor and returns the bot's response.
        The AgentExecutor will decide whether to use tools or respond directly.
        """
        # Recreate the agent/executor to ensure the system prompt reflects the latest state (e.g., hint_requested)
        # and has access to the current difficulty and topic
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
            return "I'm having trouble understanding right now. Can you please rephrase or try again."

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
