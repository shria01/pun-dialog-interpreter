# pun-dialog-interpreter
NLP dialog bot that identifies pun word senses and explains semantic humor using SBERT, WordNet, and Google Gemini API
# Overview
This project focuses on puns, a form of wordplay that relies on words with multiple meanings. Humor in puns emerges when an initial interpretation of a sentence is replaced by an unexpected but logically consistent alternative meaning. Because of this semantic incongruity, pun processing requires identifying multiple senses of words, evaluating their contextual fit, and understanding how sentence meaning shifts over time.

The goal of this project is to design a computational system capable of:

Identifying the multiple meanings involved in a pun

Explaining the semantic incongruity that produces humor

Answering user questions about a given pun

# Instructions
1. Install Python dependencies

Open PowerShell (or terminal) in your project root and run:

py -m pip install -r src/dialog_bot/requirements.txt

This will install:

gradio

google-genai

nltk

spacy

sentence-transformers

2. Download spaCy English model

This project requires the English model for spaCy. Run:

py -m spacy download en_core_web_sm

Only needs to be done once per machine.

3. Set your Google API key

Your project uses Google GenAI. In PowerShell, run:

$env:GOOGLE_API_KEY="YOUR_KEY_HERE"

Replace YOUR_KEY_HERE with your actual key.

To make it permanent:

setx GOOGLE_API_KEY "YOUR_KEY_HERE"

Keep your API key private. Do not push it to GitHub.

4. Run the chatbot

From the project root, run:

py src\dialog_bot\app.py

You should see:

Running on http://127.0.0.1:7860

Open that URL in your browser to interact with the bot.

5. NLTK setup (if needed)

If you get errors about missing NLTK data, run Python and execute:

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
exit()
6. Notes

Ensure your imports are correct:

from sense_finder.sense_finder import find_senses
from context_validator.context_validator import validate_context

Make sure you run the app from the project root (where src/ lives), so Python can find sibling modules.

This project works with the Gemini API via Google GenAI. Ensure your API key is valid and private.

