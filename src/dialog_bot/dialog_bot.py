# dialog_bot.py
# requires GEMINI_API_KEY or OPENAI_API_KEY environment variable

import os
from llm_interface.interface import LLMInterface

# Add src folder to Python path so sibling modules can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sense_finder.sense_finder import find_senses
from context_validator.context_validator import validate_context

def analyze_pun(sentence, provider: LLMInterface):
    """Analyze pun sentence using sense finder and context validator."""
    senses = find_senses(sentence)
    validation = validate_context(
        sentence, senses["pun_word"], senses["sense_a"], senses["sense_b"], provider
    )
    return {**senses, **validation}


def build_system_prompt(sentence, analysis):
    """Build the system prompt for the chat model."""
    return f"""You explain puns to people. Here's one that's been analyzed:

The sentence is: "{sentence}"
The pun word is "{analysis['pun_word']}"
First meaning: {analysis['sense_a']}
Second meaning: {analysis['sense_b']}
First meaning works in context: {analysis['sense_a_valid']}
Second meaning works in context: {analysis['sense_b_valid']}
Does the pun work: {analysis['pun_works']}
Why: {analysis['reason']}

Answer whatever the user asks about this pun. Be conversational, not robotic.
If they ask something off-topic just bring it back to the pun."""


def chat(sentence, question, history, analysis, provider: LLMInterface):
    """Generate conversational response about a pun."""
    system_prompt = build_system_prompt(sentence, analysis)

    # seed the conversation with the system prompt as a fake user/model exchange
    messages = [
        {"role": "user", "content": system_prompt + "\n\nReady to answer questions."},
        {"role": "assistant", "content": "Got it, I've reviewed the pun analysis. Ask me anything about it."},
    ]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": question})

    return provider.chat(messages)
