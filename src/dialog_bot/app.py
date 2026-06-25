# run with: python app.py

import sys
import os
import gradio as gr

# Add src folder to Python path so sibling modules can be imported
sys.path.append(os.path.dirname(__file__))

# imports from your modules
from dialog_bot import analyze_pun, chat
from llm_interface.gemini_provider import GeminiProvider
from llm_interface.openai_provider import OpenAIProvider

current_analysis = {}
current_sentence = ""

EXAMPLE_PUNS = [
    "I used to be a banker but I lost interest",
    "I'm reading a book about anti-gravity it's impossible to put down",
    "The bicycle can't stand on its own because it's two tired",
    "Time flies like an arrow fruit flies like a banana",
]

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
    global current_analysis, current_sentence
    current_sentence = sentence
    current_analysis = analyze_pun(sentence, get_provider(provider_name))

    summary = (f"**Pun word:** {current_analysis['pun_word']}\n"
               f"**Meaning A:** {current_analysis['sense_a']}\n"
               f"**Meaning B:** {current_analysis['sense_b']}\n"
               f"**Pun works:** {'Yes' if current_analysis['pun_works'] else 'No'}")
    return summary, [], []


def respond(question, history, provider_name):
    """Generate chatbot response."""
    if not current_analysis:
        msg = "Please enter a pun sentence first (above) and click Analyze."
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": msg}]
        return history, history

    pairs = to_pairs(history)
    provider = get_provider(provider_name)
    answer = chat(current_sentence, question, pairs, current_analysis, provider)
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]
    return history, history


with gr.Blocks(title="Pun Dialog Interpreter") as demo:
    gr.Markdown("# Pun Interpreter")
    gr.Markdown("Enter a pun below (or pick an example), then ask questions about it.")

    with gr.Row():
        pun_input = gr.Textbox(label="Pun sentence",
                               placeholder="Type a pun here...",
                               scale=4)
        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLE_PUNS, inputs=pun_input, label="Try one of these")

    analysis_display = gr.Markdown(label="Analysis")
    chatbot = gr.Chatbot(height=350, show_label=False)
    state = gr.State([])

    with gr.Row():
        msg_input = gr.Textbox(label="Ask a question about the pun",
                               placeholder="example: Why is this funny?",
                               scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    provider_toggle = gr.Radio(
        choices=["Gemini", "OpenAI"],
        value="Gemini",
        label="LLM Provider"
    )

    # Hook up buttons
    analyze_btn.click(fn=set_pun, inputs=[pun_input, provider_toggle],
                      outputs=[analysis_display, state, chatbot])

    send_btn.click(fn=respond, inputs=[msg_input, state, provider_toggle],
                   outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

    msg_input.submit(fn=respond, inputs=[msg_input, state, provider_toggle],
                     outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
