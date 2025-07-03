from socrabot_logic_v3 import SocraticBot
from config import load_environment_variables
from langchain_core.messages import HumanMessage, AIMessage
from logger import setup_logging # Import the setup_logging function

def main():
    """
    Main function to run the Socratic Python Tutor in a console interface.
    """
    # Initializing logger
    logger = setup_logging()
    logger.info("Socratic Bot session started.")

    google_api_key = load_environment_variables()
    bot = SocraticBot(api_key=google_api_key) # type: ignore

    print("=" * 30)
    print("Socratic Python Tutor (Console Demo)")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'hint' for a clue.")
    print("Type 'easier' or 'harder' to adjust difficulty.")
    print("Just paste your Python code directly for review anytime.")
    print("Type 'I understand' or 'challenge me' for a fill-in-the-blanks challenge.")
    print("=" * 30)
    logger.info("Displayed initial instructions to user.")

    initial_topic = "variables in Python"
    bot.update_current_topic(initial_topic)

    welcome_message = f"Hello! I'm your Socratic Python Tutor. Today, we can start with '{initial_topic}'."
    print(f"Bot: {welcome_message}")
    bot.add_message_to_history(AIMessage(content=welcome_message))
    logger.info(f"Bot: {welcome_message}")

    # Give options to the user
    options_message = "\nBot: Would you like to:\n1. Test your knowledge on variables in Python?\n2. Learn more about variables in Python?\nPlease type '1' or '2'."
    print(options_message)
    logger.info(options_message)

    first_choice_made = False

    while True:
        user_input = input("You: ").strip()
        logger.info(f"User: {user_input}")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! Keep coding!")
            logger.info("User exited the session. Goodbye!")
            break

        if not first_choice_made:
            if user_input == '1':
                response_text = f"Great! Let's test your knowledge on {bot.current_topic}."
                print(f"Bot: {response_text}")
                logger.info(f"Bot: {response_text} (User chose to test knowledge)")
                bot.add_message_to_history(HumanMessage(content="user chose to test knowledge"))
                
                # Directly call the challenge tool
                challenge_text = bot.send_message_to_llm(
                    f"Generate a challenge for '{bot.current_topic}'.",
                    hint_requested=False # This is not a hint request
                )
                print(f"Bot: {challenge_text}")
                logger.info(f"Bot (Challenge): {challenge_text}")
                bot.add_message_to_history(AIMessage(content=challenge_text))
                first_choice_made = True
                continue
            elif user_input == '2':
                response_text = f"Excellent! Let's dive deeper into {bot.current_topic}."
                print(f"Bot: {response_text}")
                logger.info(f"Bot: {response_text} (User chose to learn more)")
                bot.add_message_to_history(HumanMessage(content="user chose to learn more"))
                
                # Continue with initial Socratic question
                initial_socratic_q = f"What have you learned so far, or what are you curious about regarding {bot.current_topic}?"
                print(f"Bot: {initial_socratic_q}")
                logger.info(f"Bot: {initial_socratic_q}")
                bot.add_message_to_history(AIMessage(content=initial_socratic_q))
                first_choice_made = True
                continue
            else:
                print("Bot: Please choose '1' to test your knowledge or '2' to learn more.")
                logger.warning(f"User entered invalid initial choice: {user_input}")
                continue

        # Normal conversation flow after the initial choice
        try:
            if user_input.lower() == "hint":
                print("Bot: Let me give you a small hint...")
                logger.info("User requested a hint.")
                bot.add_message_to_history(HumanMessage(content="hint requested"))
                bot_response_text = bot.send_message_to_llm("I need a hint for the current topic.", hint_requested=True)
                print(f"Bot: {bot_response_text}")
                logger.info(f"Bot (Hint): {bot_response_text}")
                bot.add_message_to_history(AIMessage(content=bot_response_text))
                
            elif user_input.lower() == "easier":
                bot.set_difficulty(bot.get_difficulty() - 1)
                response_text = f"Okay, I've adjusted the difficulty to easier (Level: {bot.get_difficulty()}). How about we review the basics of Python data types?"
                print(f"Bot: {response_text}")
                logger.info(f"Bot: {response_text} (Difficulty set to Easier)")
                bot.add_message_to_history(HumanMessage(content=user_input))
                bot.add_message_to_history(AIMessage(content=response_text))
                bot.update_current_topic("basic Python data types")
                continue
            elif user_input.lower() == "harder":
                bot.set_difficulty(bot.get_difficulty() + 1)
                response_text = f"Great! I've adjusted the difficulty to harder (Level: {bot.get_difficulty()}). Can you explain the difference between mutable and immutable data types in Python?"
                print(f"Bot: {response_text}")
                logger.info(f"Bot: {response_text} (Difficulty set to Harder)")
                bot.add_message_to_history(HumanMessage(content=user_input))
                bot.add_message_to_history(AIMessage(content=response_text))
                bot.update_current_topic("mutable and immutable data types")
                continue
            elif not user_input:
                print("Please type something.")
                logger.warning("User entered empty input.")
                continue
            else:
                # For regular input - text or code (I understand, challenge me)
                bot.add_message_to_history(HumanMessage(content=user_input))
                bot_response_text = bot.send_message_to_llm(user_input, hint_requested=False)
                print(f"Bot: {bot_response_text}")
                logger.info(f"Bot: {bot_response_text}")
                
                # Adjusting difficulty based on the bot's response
                bot.adjust_difficulty_based_on_response(bot_response_text, False) 

        except Exception as e:
            print(f"Bot: An unexpected error occurred: {e}. Please try again.")
            logger.exception("An unexpected error occurred in the main loop.") # Logs full traceback

if __name__ == "__main__":
    main()