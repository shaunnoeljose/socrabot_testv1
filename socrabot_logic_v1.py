import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

class SocraticBot:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Initializes the SocraticBot with a Gemini LLM.

        Args:
            api_key (str): The Google API key for Gemini.
            model_name (str): The name of the Gemini model to use.
            temperature (float): The sampling temperature for the LLM.
        """
        genai.configure(api_key=api_key) # type: ignore
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.chat_history = [] # Stores HumanMessage and AIMessage objects
        self.difficulty = 0    # Initial difficulty level (0: Novice)

    def _generate_socratic_prompt(self, user_message: str, hint_requested: bool = False) -> str:
        """
        Generates a socratic system prompt for the LLM based on user message,
        current difficulty, and whether a hint was requested.
        """
        difficulty_hint = ""
        if self.difficulty == 0:
            difficulty_hint = "Keep your questions simple, foundational, and guide the user gently. Provide clear examples. Break down complex ideas into smaller, manageable parts."
        elif self.difficulty == 1:
            difficulty_hint = "Ask slightly more challenging questions, prompting deeper thought but still offering clear pathways. Use follow-up questions to probe understanding and connect concepts."
        elif self.difficulty == 2:
            difficulty_hint = "Pose challenging, open-ended questions that require more critical thinking and problem-solving. Encourage independent research or experimentation. Focus on nuanced understanding."
        else:
            difficulty_hint = "Be a helpful Socratic tutor."

        hint_instruction = ""
        if hint_requested:
            hint_instruction = "The user has explicitly asked for a hint. Provide a subtle clue or a guiding question that helps them move forward without giving away the direct answer. Ensure it's concise."

        # The prompt template guides the LLM to act as a Socratic tutor.
        # It emphasizes asking questions, avoiding direct answers, and efficient learning.
        return (
            f"You are a Socratic Python programming tutor for novice learners. "
            f"Your primary goal is to guide the user to discover solutions and understand concepts through questioning, "
            f"aiming for efficient learning and quick grasp of concepts. "
            f"Do not provide direct answers or code solutions. "
            f"Always ask a follow-up question unless the user explicitly requests to exit. "
            f"{difficulty_hint} {hint_instruction} "
            f"The user just said: \"{user_message}\""
        )

    def send_message_to_llm(self, user_message: str, hint_requested: bool = False) -> str:
        """
        Sends the user's message to the LLM and returns the bot's response.
        Manages chat history and applies socratic prompting.
        """
        # Generate the Socratic prompt for the current user input
        socratic_system_prompt = self._generate_socratic_prompt(user_message, hint_requested)

        # Create the full prompt template including chat history
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", socratic_system_prompt),
            MessagesPlaceholder(variable_name="chat_history")
        ])

        # Chain the prompt with the LLM
        chain = prompt_template | self.llm

        try:
            # Invoke the chain with the current chat history
            response = chain.invoke({"chat_history": self.chat_history})
            bot_response_text = response.content
            return bot_response_text # type: ignore
        except Exception as e:
            print(f"Error communicating with LLM: {e}")
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




# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api = google_api_key)

# # --- Socratic Prompt Generation ---
# def generate_socratic_prompt(user_message: str, current_difficulty: int) -> str:
#     """
#     Generates a socratic prompt for the LLM based on user message and difficulty.
#     The difficulty is a simple integer (0: Novice, 1: Beginner, 2: Intermediate).
#     """
#     difficulty_hint = ""
#     if current_difficulty == 0:
#         difficulty_hint = "Keep your questions simple, foundational, and guide the user gently. Provide clear examples."
#     elif current_difficulty == 1:
#         difficulty_hint = "Ask slightly more challenging questions, prompting deeper thought but still offering clear pathways. Use follow-up questions to probe understanding."
#     elif current_difficulty == 2:
#         difficulty_hint = "Pose challenging, open-ended questions that require more critical thinking and problem-solving. Encourage independent research or experimentation."
#     else:
#         difficulty_hint = "Be a helpful Socratic tutor."

#     # The prompt template guides the LLM to act as a Socratic tutor.
#     # It emphasizes asking questions and avoiding direct answers.
#     return (
#         f"You are a Socratic Python programming tutor for novice learners. "
#         f"Your goal is to guide the user to discover solutions and understand concepts through questioning, not by directly providing answers. "
#         f"Always ask a follow-up question. Do not provide code solutions directly. "
#         f"{difficulty_hint} "
#         f"The user just said: \"{user_message}\""
#     )

# # --- Chat Functionality ---
# def run_socratic_bot():
#     """
#     Runs the Socratic Python Tutor in a console interface.
#     """
#     print("=" * 50)
#     print("Socratic Python Tutor (Console Demo)")
#     print("Type 'exit' or 'quit' to end the session.")
#     print("=" * 50)

#     chat_history = [] # Stores HumanMessage and AIMessage objects
#     difficulty = 0    # Initial difficulty level (0: Novice)

#     # Initial welcome message from the bot
#     initial_bot_message = "Hello! I'm your Socratic Python Tutor. Let's start with a basic Python concept. What do you know about variables in Python?"
#     print(f"Bot: {initial_bot_message}")
#     chat_history.append(AIMessage(content=initial_bot_message))

#     while True:
#         user_input = input("You: ").strip()

#         if user_input.lower() in ["exit", "quit"]:
#             print("Bot: Goodbye! Keep coding!")
#             break

#         if not user_input:
#             print("Please type something.")
#             continue

#         # Add user message to chat history
#         chat_history.append(HumanMessage(content=user_input))

#         # Generate the Socratic prompt for the current user input
#         socratic_prompt = generate_socratic_prompt(user_input, difficulty)

#         # Create the full prompt template including chat history
#         # MessagesPlaceholder allows including previous messages in the conversation.
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", socratic_prompt),
#             MessagesPlaceholder(variable_name="chat_history")
#         ])

#         # Chain the prompt with the LLM
#         chain = prompt_template | llm

#         print("Bot is thinking...")
#         try:
#             # Invoke the chain with the current chat history
#             response = chain.invoke({"chat_history": chat_history})
#             bot_response_text = response.content
#             print(f"Bot: {bot_response_text}")

#             # Add bot response to chat history
#             chat_history.append(AIMessage(content=bot_response_text))

#             # --- Simple Learner Model for Difficulty Adjustment ---
#             # This is a very basic heuristic; in a real application,
#             # this would be based on code analysis, problem-solving success, etc.
#             if len(bot_response_text) < 100: # If bot response is short, implies user is on track, increase difficulty
#                 difficulty = min(difficulty + 1, 2) # Max difficulty 2
#             elif len(bot_response_text) > 200: # If bot response is long, implies user needs more guidance, decrease difficulty
#                 difficulty = max(difficulty - 1, 0) # Min difficulty 0
#             # print(f"(Debug: Current difficulty level: {difficulty})") # Uncomment for debugging difficulty changes

#         except Exception as e:
#             print(f"Bot: An error occurred while generating a response: {e}")
#             print("Please try again.")
#             # If an error occurs, we don't want to add an empty response to history,
#             # but keep the user's last message for context in next attempt.

# if __name__ == "__main__":
#     run_socratic_bot()
