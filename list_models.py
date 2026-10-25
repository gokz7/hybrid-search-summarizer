# list_models.py
# Helper: prints every Gemini model your API key can use for text generation.
# Use it to pick GEMINI_MODEL and GEMINI_FALLBACK_MODELS in config.py
# (model names change often, so check here instead of guessing).
# Run:  python list_models.py

import os                         # read the API key
from dotenv import load_dotenv    # load .env
from google import genai          # Gemini SDK

load_dotenv()                                             # GEMINI_API_KEY from .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))   # connect
for model in client.models.list():                        # every model visible to this key
    actions = model.supported_actions or []               # what the model can do
    if "generateContent" in actions:                      # only models that can write text
        print(model.name.replace("models/", ""))          # print the short name, ready for config.py