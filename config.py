import os
from dotenv import load_dotenv

def load_config():
    if os.path.exists("debug.env"):
        print("DEBUG MODE ACTIVATED")
        load_dotenv("debug.env")
    else:
        print("PRODUCTION MODE ACTIVATED")