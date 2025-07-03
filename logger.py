# logger.py
import logging
import os
from datetime import datetime

def setup_logging(log_file_name="socratic_bot.log", level=logging.INFO):
    """
    Sets up a logger that outputs messages to a specified file.

    Args:
        log_file_name (str): The name of the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create a 'logs' directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name)

    # Get the root logger
    logger = logging.getLogger('socratic_bot') # Give your logger a specific name
    logger.setLevel(level)
    logger.propagate = False # Prevent messages from being passed to ancestor loggers

    # Clear existing handlers to prevent duplicate output if called multiple times
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.handlers = [] # Ensure list is empty

    # Create a file handler
    file_handler = logging.FileHandler(log_path, mode='a') # 'a' for append mode
    file_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Removed console handler to stop printing logs to console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return logger

# Example usage (you would remove this block in a real application,
# but it shows how to use the logger)
if __name__ == "__main__":
    logger = setup_logging(level=logging.DEBUG) # Set to DEBUG to see all messages

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        logger.exception("An error occurred during division.") # logs exception info automatically

    print(f"\nLog file created at: logs/socratic_bot.log")
    print("Check the 'logs' directory for the log file. No console output from the logger here.")
