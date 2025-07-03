import os
from dotenv import load_dotenv

def load_environment_variables():
    """
    Loads the environment variables from .env file
    """
    
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
        
    return google_api_key