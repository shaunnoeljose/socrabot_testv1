import os
from dotenv import load_dotenv

def load_environment_variables():

    """Loads the environment variables from .env file."""
    
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("Error: GOOGLE_API_KEY not present in .env file .")
        exit(1) 
        
    return google_api_key