import time

from llm_interface.interface import LLMInterface
from sense_finder.sense_finder import (
    parse_sentence,
    rank_sense_candidates,
    retrieve_wordnet_candidates,
)
from context_validator.context_validator import validate_context


def parse_pun_sentence(sentence):
    return parse_sentence(sentence)


def retrieve_candidates(doc):
    return retrieve_wordnet_candidates(doc)


def retrieve_senses(sentence, candidates):
    return rank_sense_candidates(sentence, candidates)


def validate_candidates(sentence, candidates, provider):
    return validate_context(sentence, candidates, provider)


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
Keep the answer concise and under 120 words.
If they ask something off-topic just bring it back to the pun."""


def chat(sentence, question, history, analysis, provider: LLMInterface):
    """Generate conversational response about a pun."""
    system_prompt = build_system_prompt(sentence, analysis)

    # seed the conversation with the system prompt as a fake user/model exchange
    messages = [
        {"role": "user", "content": system_prompt + "\n\nReady to answer questions."},
        {"role": "assistant", "content": "Got it, I've reviewed the pun analysis. Ask me anything about it."},
    ]

    # Keep prompts fast and bounded as a conversation grows.
    for user_msg, bot_msg in history[-3:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": question})

    started = time.perf_counter()
    response = provider.chat(messages)
    print(f"timing chat_request={time.perf_counter() - started:.3f}s", flush=True)
    return response
