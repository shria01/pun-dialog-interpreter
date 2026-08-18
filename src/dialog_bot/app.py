# run with: python app.py

import sys
import os
import gradio as gr

# Add src folder to Python path so sibling modules can be imported
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(SRC_DIR)

# imports from your modules
from dialog_bot import analyze_pun, chat
from llm_interface.gemini_provider import GeminiProvider
from llm_interface.openai_provider import OpenAIProvider

EXAMPLE_PUNS = [
    "I used to be a banker but I lost interest",
    "The math teacher was a good ruler",
    "A boiled egg in the morning is hard to beat",
    "I used to hate facial hair but then it grew on me",
    "Broken pencils are pointless",
]

AVAILABLE_PROVIDERS = []
if os.environ.get("GEMINI_API_KEY"):
    AVAILABLE_PROVIDERS.append("Gemini")
if os.environ.get("OPENAI_API_KEY"):
    AVAILABLE_PROVIDERS.append("OpenAI")

# Keep the UI renderable when no key is configured so the resulting error
# clearly points the local developer to the missing default key.
if not AVAILABLE_PROVIDERS:
    AVAILABLE_PROVIDERS = ["Gemini"]

def get_provider(name):
    if name == "Gemini":
        return GeminiProvider()
    elif name == "OpenAI":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")

def to_pairs(history):
    """Convert Gradio history dicts into (user, bot) tuples."""
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def set_pun(sentence, provider_name):
    """Analyze pun sentence and return summary."""
    analysis = analyze_pun(sentence, get_provider(provider_name))

    summary = (f"**Pun word:** {analysis['pun_word']}\n"
               f"**Meaning A:** {analysis['sense_a']}\n"
               f"**Meaning B:** {analysis['sense_b']}\n"
               f"**Pun works:** {'Yes' if analysis['pun_works'] else 'No'}")
    session = {"sentence": sentence, "analysis": analysis}
    return summary, [], [], session


def respond(question, history, provider_name, session):
    """Generate chatbot response."""
    analysis = session.get("analysis") if session else None
    sentence = session.get("sentence") if session else None
    if not analysis or not sentence:
        msg = "Please enter a pun sentence first (above) and click Analyze."
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": msg}]
        return history, history

    pairs = to_pairs(history)
    provider = get_provider(provider_name)
    answer = chat(sentence, question, pairs, analysis, provider)
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]
    return history, history


with gr.Blocks(title="Pun Dialog Interpreter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Pun Interpreter")
    gr.Markdown("Enter a pun below (or pick an example), then ask questions about it.")

    with gr.Row():
        pun_input = gr.Textbox(label="Pun sentence",
                               placeholder="Type a pun here...",
                               scale=4)
        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLE_PUNS, inputs=pun_input, label="Try one of these")

    analysis_display = gr.Markdown(label="Analysis")
    chatbot = gr.Chatbot(height=350, show_label=False, type="messages")
    state = gr.State([])
    session_state = gr.State({"sentence": "", "analysis": None})

    with gr.Row():
        msg_input = gr.Textbox(label="Ask a question about the pun",
                               placeholder="example: Why is this funny?",
                               scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    provider_toggle = gr.Radio(
        choices=AVAILABLE_PROVIDERS,
        value=AVAILABLE_PROVIDERS[0],
        label="LLM Provider"
    )

    # Hook up buttons
    analyze_btn.click(fn=set_pun, inputs=[pun_input, provider_toggle],
                      outputs=[analysis_display, state, chatbot, session_state])

    send_btn.click(fn=respond, inputs=[msg_input, state, provider_toggle, session_state],
                   outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

    msg_input.submit(fn=respond, inputs=[msg_input, state, provider_toggle, session_state],
                     outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "8080")),
    )
