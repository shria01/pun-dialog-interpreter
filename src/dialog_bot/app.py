import os
import sys
import time
from functools import lru_cache

import gradio as gr

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(SRC_DIR)

from dialog_bot import (
    chat,
    parse_pun_sentence,
    retrieve_candidates,
    retrieve_senses,
    validate_candidates,
)
from llm_interface.gemini_provider import GeminiProvider
from llm_interface.openai_provider import OpenAIProvider


EXAMPLE_PUNS = {
    "Lost interest": "I used to be a banker but I lost interest",
    "Pointless": "Broken pencils are pointless",
    "Grew on me": "I used to hate facial hair but then it grew on me",
}
FOLLOW_UPS = [
    "Why is this funny?",
    "Which meaning is literal?",
    "Could you explain the wordplay?",
]

AVAILABLE_PROVIDERS = []
if os.environ.get("GEMINI_API_KEY"):
    AVAILABLE_PROVIDERS.append("Gemini")
if os.environ.get("OPENAI_API_KEY"):
    AVAILABLE_PROVIDERS.append("OpenAI")
if not AVAILABLE_PROVIDERS:
    AVAILABLE_PROVIDERS = ["Gemini"]

PUN_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0b12",
    block_background_fill="#16182a",
    block_border_color="#23263a",
    body_text_color="#d5d7e2",
    body_text_color_subdued="#9a9fb5",
    button_primary_background_fill="#7f77dd",
    button_primary_background_fill_hover="#948cf0",
    button_primary_text_color="#ffffff",
)

APP_CSS = """
:root {
  --bg-page: #0a0b12;
  --bg-card: #16182a;
  --bg-card-alt: #1a1622;
  --border: #23263a;
  --border-strong: #2e3148;
  --text-primary: #f4f5f9;
  --text-body: #d5d7e2;
  --text-secondary: #9a9fb5;
  --text-muted: #6b6f80;
  --accent: #7f77dd;
  --accent-hover: #948cf0;
  --literal: #a9a5ec;
  --figurative: #e0b35a;
  --success: #63a67a;
}
.gradio-container footer { display: none !important; }
#app-shell {
  max-width: 1040px;
  margin: 0 auto;
  padding: 36px 24px 48px;
}
.hero { margin-bottom: 22px; }
.hero p:first-child, .anchor-label p {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 650;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.hero h1 {
  color: var(--text-primary) !important;
  font-size: 42px !important;
  line-height: 1.08;
  margin: 8px 0 10px !important;
}
.hero p:last-child { color: var(--text-body); font-size: 17px; }
.field-label p { color: var(--text-body); font-size: 14px; font-weight: 600; margin-bottom: 6px; }
.input-card {
  background: #111426 !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  gap: 10px !important;
  padding: 18px 20px !important;
}
.input-card .field-label.block,
.input-card .example-row,
.input-card #input-row {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
.input-card .example-label p {
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  margin: 2px 0 0;
}
#pun-input textarea, #question-input textarea {
  background: var(--bg-card) !important;
  border-color: var(--border-strong) !important;
  color: var(--text-primary) !important;
  height: 52px !important;
  min-height: 52px !important;
}
#pun-input {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
#analyze-button, #ask-button {
  background: var(--accent) !important;
  color: #fff !important;
  min-height: 52px;
  font-weight: 650;
}
#analyze-button:hover, #ask-button:hover { background: var(--accent-hover) !important; }
.example-row, .followup-row { gap: 8px; flex-wrap: wrap !important; }
.example-chip, .followup-chip {
  background: var(--bg-card) !important;
  border: 0 !important;
  border-radius: 999px !important;
  color: var(--text-secondary) !important;
  min-width: auto !important;
}
.input-card .example-chip {
  min-height: 44px !important;
  padding: 8px 12px !important;
  white-space: normal !important;
}
.example-chip:hover, .followup-chip:hover {
  background: var(--border-strong) !important;
  color: var(--text-primary) !important;
}
.work-status.block {
  background: transparent !important;
  border: 0 !important;
  padding: 12px 2px 0 !important;
}
.work-status p {
  align-items: center;
  color: var(--text-secondary);
  display: flex;
  font-size: 13px;
  gap: 10px;
  margin: 0;
}
.work-status strong { color: var(--text-primary); }
.work-status strong:first-of-type::after {
  animation: analysis-dots 1.1s ease-in-out infinite;
  background: radial-gradient(circle, var(--accent) 1.7px, transparent 2px) 0 50% / 6px 6px repeat-x;
  content: "";
  display: inline-block;
  height: 7px;
  margin-left: 6px;
  transform-origin: left center;
  width: 18px;
}
@keyframes analysis-dots {
  0%, 100% { opacity: .3; }
  50% { opacity: 1; }
}
.result-divider hr { border-color: var(--border); margin: 34px 0 28px; }
.analysis-result .anchor-label p { margin: 0 0 8px; }
.result-heading { align-items: center; margin: 0 0 18px; }
.pun-word h2 {
  color: var(--text-primary) !important;
  font-size: 34px !important;
  font-weight: 700 !important;
  letter-spacing: -.01em;
  line-height: 1.12;
  margin: 0 !important;
}
.status p {
  color: var(--success);
  font-size: 13px;
  font-weight: 650;
  white-space: nowrap;
  text-align: right;
}
.meaning-row { gap: 16px; margin-bottom: 16px; }
.meaning-card.block {
  background: var(--bg-card);
  border: 1px solid var(--border-strong);
  border-radius: 10px;
  min-height: 138px;
  overflow: hidden;
  padding: 18px 20px;
  position: relative;
}
.meaning-card.secondary.block {
  background: var(--bg-card-alt);
}
#sense-a-card::before,
#sense-b-card::before {
  bottom: 0;
  content: "";
  left: 0;
  position: absolute;
  top: 0;
  width: 4px;
  z-index: 2;
}
#sense-a-card::before { background: var(--literal); }
#sense-b-card::before { background: var(--figurative); }
.meaning-card h4 {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  margin: 0 0 12px !important;
}
.meaning-card p { color: var(--text-body); font-size: 14px; line-height: 1.6; }
.why-panel.block {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 17px 20px;
}
.why-panel h4 {
  color: var(--text-primary) !important;
  font-size: 14px !important;
  margin: 0 0 8px !important;
}
.why-panel p {
  color: var(--text-body);
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 0;
  max-width: 780px;
}
.provider-note p { color: var(--text-muted); font-size: 11px; margin: 10px 0 0; }
.chat-shell {
  background: #111426 !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 18px 20px !important;
}
.chat-shell .chat-heading.block,
.chat-shell .chat-header,
.chat-shell .followup-row,
.chat-shell #question-row {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
.chat-header { align-items: flex-start !important; gap: 16px !important; }
.chat-heading h3 { color: var(--text-primary) !important; font-size: 18px !important; margin: 0 0 3px !important; }
.chat-heading p { color: var(--text-secondary); font-size: 13px; margin: 0 0 8px; }
#provider-toggle {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 999px !important;
  flex: 0 0 auto !important;
  margin-left: auto !important;
  max-width: none !important;
  min-width: 0 !important;
  padding: 3px !important;
  width: fit-content !important;
}
#provider-toggle > div { gap: 2px !important; width: fit-content !important; }
#provider-toggle label {
  background: transparent !important;
  border: 0 !important;
  border-radius: 999px !important;
  color: var(--text-secondary) !important;
  min-width: 88px !important;
  padding: 6px 10px !important;
}
#provider-toggle input { display: none !important; }
#provider-toggle label:has(input:checked) {
  background: var(--accent) !important;
  color: #fff !important;
}
#pun-chatbot {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  margin: 0 0 6px;
}
#pun-chatbot .bubble-wrap,
#pun-chatbot .panel-wrap,
#pun-chatbot .message-wrap {
  background: transparent !important;
}
#pun-chatbot .bubble-wrap,
#pun-chatbot .panel-wrap {
  scrollbar-color: var(--border-strong) transparent;
  scrollbar-width: thin;
}
#pun-chatbot .bubble-wrap::-webkit-scrollbar,
#pun-chatbot .panel-wrap::-webkit-scrollbar {
  width: 8px;
}
#pun-chatbot .bubble-wrap::-webkit-scrollbar-track,
#pun-chatbot .panel-wrap::-webkit-scrollbar-track {
  background: transparent;
}
#pun-chatbot .bubble-wrap::-webkit-scrollbar-thumb,
#pun-chatbot .panel-wrap::-webkit-scrollbar-thumb {
  background: #565b78;
  border: 2px solid #111426;
  border-radius: 999px;
}
#pun-chatbot .bubble-wrap::-webkit-scrollbar-thumb:hover,
#pun-chatbot .panel-wrap::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}
#pun-chatbot button[aria-label="Clear"],
#pun-chatbot button[title="Clear"] { display: none !important; }
#pun-chatbot .message,
#pun-chatbot .message p,
#pun-chatbot .message li {
  font-size: 12.5px !important;
  line-height: 1.5 !important;
}
#pun-chatbot .message-row.bubble {
  margin: 6px 10px !important;
  max-width: 86% !important;
}
#pun-chatbot .message-row.user-row { margin-left: auto !important; }
#pun-chatbot .message-row.bot-row { margin-right: auto !important; }
#pun-chatbot .message-row > .flex-wrap.user,
#pun-chatbot .message-row > .flex-wrap.bot {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  height: auto !important;
  padding: 0 !important;
  width: 100% !important;
}
#pun-chatbot .message-row > .flex-wrap > .message,
#pun-chatbot .message-row > .flex-wrap > .message > .message {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
  padding: 0 !important;
  width: auto !important;
}
#pun-chatbot [data-testid="user"],
#pun-chatbot [data-testid="bot"] {
  box-sizing: border-box;
  display: block;
  max-width: 100%;
  padding: 8px 12px !important;
  width: fit-content;
}
#pun-chatbot [data-testid="user"] {
  background: var(--accent) !important;
  border-radius: 16px 16px 5px 16px !important;
  color: white !important;
  margin-left: auto;
}
#pun-chatbot [data-testid="bot"] {
  background: var(--bg-card-alt) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 16px 16px 16px 5px !important;
  color: var(--text-body) !important;
  margin-right: auto;
}
#pun-chatbot [data-testid="bot"][aria-label*="Thinking with"]::after {
  animation: thinking-pulse 1s ease-in-out infinite;
  background: radial-gradient(circle, currentColor 1.6px, transparent 2px) 0 50% / 6px 6px repeat-x;
  content: "";
  display: inline-block;
  height: 6px;
  margin-left: 7px;
  width: 18px;
}
@keyframes thinking-pulse {
  0%, 100% { opacity: .3; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(-1px); }
}
#pun-chatbot .message p { margin: 0 0 6px !important; }
#pun-chatbot .message p:last-child { margin-bottom: 0 !important; }
.chat-shell .followup-row { margin: 2px 0 10px; }
.chat-shell .followup-chip {
  background: #24283d !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-body) !important;
  font-size: 12.5px !important;
}
.chat-shell .followup-chip:hover {
  background: #2b3048 !important;
  border-color: var(--accent) !important;
  color: var(--text-primary) !important;
}
#question-row { gap: 10px; margin-top: 2px; }
#question-input { background: var(--bg-card) !important; border: 1px solid var(--border-strong) !important; }
#question-input textarea { background: transparent !important; border: 0 !important; }
.bottom-meta { margin-top: 28px; }
.bottom-meta p { color: var(--text-secondary); font-size: 12px; }
@media (max-width: 700px) {
  #app-shell { padding: 26px 16px 36px; }
  .meaning-row, .chat-header, #input-row, #question-row { flex-direction: column; }
  .result-heading { align-items: flex-start; flex-direction: column; }
  .status p { text-align: left; }
  #provider-toggle { margin-left: 0 !important; }
  #analyze-button, #ask-button { width: 100%; }
}
"""


@lru_cache(maxsize=2)
def get_provider(name):
    if name == "Gemini":
        return GeminiProvider()
    if name == "OpenAI":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")


def to_pairs(history):
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def chat_greeting():
    return [{
        "role": "assistant",
        "content": "✦ Pun decoded. Curious about either interpretation? Ask me anything.",
    }]


def analysis_values(analysis, provider_name):
    word = str(analysis.get("pun_word", "No clear pun"))
    sense_a = str(analysis.get("sense_a", "No first interpretation found."))
    sense_b = str(analysis.get("sense_b", "No second interpretation found."))
    reason = str(analysis.get("reason", "No explanation was returned."))
    works = bool(analysis.get("pun_works"))
    status = "✓ Double meaning confirmed" if works else "No clear pun detected"
    return (
        f"## {word}", status,
        f"#### Interpretation A\n\n{sense_a}",
        f"#### Interpretation B\n\n{sense_b}",
        f"#### Why it works\n\n{reason}",
        f"spaCy + WordNet + SBERT analysis · {provider_name} validation and explanation",
    )


def render_error(provider_name, error):
    message = (
        "Try a sentence containing a word with two possible meanings."
        if isinstance(error, ValueError)
        else f"The {provider_name} request could not be completed. Try again or switch models."
    )
    return f"### Couldn't analyze this sentence\n\n{message}"


def analyze_with_progress(sentence, provider_name):
    """Report real pipeline boundaries while analysis is running."""
    unchanged = [gr.update() for _ in range(14)]
    yield (
        *unchanged,
        gr.update(interactive=False),
        gr.update(
            value=(
                "**Analyzing pun** · **Parsing with spaCy** · "
                "Retrieving WordNet senses · Ranking with SBERT · "
                f"Validating with {provider_name}"
            ),
            visible=True,
        ),
    )
    sentence = sentence.strip()
    started = time.perf_counter()
    try:
        if not sentence:
            raise ValueError("Empty sentence")

        doc = parse_pun_sentence(sentence)
        yield (
            *unchanged,
            gr.update(interactive=False),
            gr.update(
                value=(
                    "**Analyzing pun** · Parsing with spaCy · "
                    "**Retrieving WordNet senses** · Ranking with SBERT · "
                    f"Validating with {provider_name}"
                ),
                visible=True,
            ),
        )

        raw_candidates = retrieve_candidates(doc)
        yield (
            *unchanged,
            gr.update(interactive=False),
            gr.update(
                value=(
                    "**Analyzing pun** · Parsing with spaCy · "
                    "Retrieving WordNet senses · **Ranking with SBERT** · "
                    f"Validating with {provider_name}"
                ),
                visible=True,
            ),
        )

        candidates = retrieve_senses(sentence, raw_candidates)
        yield (
            *unchanged,
            gr.update(interactive=False),
            gr.update(
                value=(
                    "**Analyzing pun** · Parsing with spaCy · "
                    "Retrieving WordNet senses · Ranking with SBERT · "
                    f"**Validating with {provider_name}**"
                ),
                visible=True,
            ),
        )

        provider = get_provider(provider_name)
        analysis = validate_candidates(sentence, candidates, provider)
        session = {"sentence": sentence, "analysis": analysis}
        greeting = chat_greeting()
        result = (
            *analysis_values(analysis, provider_name),
            gr.update(value="", visible=False),
            greeting,
            gr.update(value=greeting, visible=True, height=140),
            session,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except Exception as error:
        result = (
            "", "", "", "", "", "",
            gr.update(value=render_error(provider_name, error), visible=True),
            [],
            gr.update(value=[], visible=False),
            {"sentence": "", "analysis": None},
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    print(f"timing analyze_total={time.perf_counter() - started:.3f}s", flush=True)
    yield (
        *result,
        gr.update(value="Analyze pun →", interactive=True),
        gr.update(visible=False),
    )


def respond(question, history, provider_name, session):
    question = question.strip()
    analysis = session.get("analysis") if session else None
    sentence = session.get("sentence") if session else None
    if not question:
        return gr.update(), history
    if not analysis or not sentence:
        answer = "Analyze a pun first, then ask a question about its wordplay."
    else:
        try:
            answer = chat(sentence, question, to_pairs(history), analysis, get_provider(provider_name))
        except Exception:
            answer = f"The {provider_name} request couldn't be completed. Try again or switch models."
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return gr.update(value=history, visible=True, height=240), history


def respond_with_progress(question, history, provider_name, session):
    """Show the question and a temporary bot bubble before the final answer."""
    question = question.strip()
    pending = list(history)
    if question:
        pending.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"✦ Thinking with {provider_name}"},
        ])
    yield (
        gr.update(value=pending, visible=True, height=240),
        gr.update(),
        gr.update(value="Generating…", interactive=False),
        "",
    )
    thread, updated_history = respond(question, history, provider_name, session)
    yield (
        thread,
        updated_history,
        gr.update(value="Ask →", interactive=True),
        "",
    )


def suggestion_handler(question):
    def handle(history, provider_name, session):
        yield from respond_with_progress(question, history, provider_name, session)
    return handle


with gr.Blocks(title="Pun Interpreter", theme=PUN_THEME, css=APP_CSS) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.Markdown(
            "NLP · WORD SENSE DISAMBIGUATION\n\n# Pun Interpreter\n\n"
            "Decode the double meaning behind a pun.",
            elem_classes="hero",
        )
        with gr.Column(elem_classes="input-card"):
            gr.Markdown("Enter a pun", elem_classes="field-label")
            with gr.Row(elem_id="input-row"):
                pun_input = gr.Textbox(
                    label=None, placeholder="Broken pencils are pointless",
                    lines=1, scale=5, show_label=False, elem_id="pun-input",
                )
                analyze_btn = gr.Button(
                    "Analyze pun →", variant="primary", scale=1, elem_id="analyze-button",
                )
            gr.Markdown("Or choose an example", elem_classes=["field-label", "example-label"])
            with gr.Row(elem_classes="example-row"):
                example_buttons = [
                    gr.Button(
                        sentence,
                        size="sm",
                        variant="secondary",
                        elem_classes="example-chip",
                    )
                    for sentence in EXAMPLE_PUNS.values()
                ]
        analysis_progress = gr.Markdown(visible=False, elem_classes="work-status")

        analysis_group = gr.Column(visible=False)
        with analysis_group:
            gr.Markdown("---", elem_classes="result-divider")
            error_display = gr.Markdown(visible=False, elem_classes="error-card")

            result_group = gr.Column(
                visible=False,
                elem_classes=["result-reveal", "analysis-result"],
            )
            with result_group:
                gr.Markdown("ANALYSIS", elem_classes="anchor-label")
                with gr.Row(elem_classes="result-heading"):
                    with gr.Column(scale=4):
                        word_display = gr.Markdown(elem_classes="pun-word")
                    with gr.Column(scale=1):
                        status_display = gr.Markdown(elem_classes="status")
                with gr.Row(elem_classes="meaning-row"):
                    with gr.Column(scale=1):
                        sense_a_display = gr.Markdown(
                            elem_id="sense-a-card",
                            elem_classes="meaning-card",
                        )
                    with gr.Column(scale=1):
                        sense_b_display = gr.Markdown(
                            elem_id="sense-b-card",
                            elem_classes=["meaning-card", "secondary"],
                        )
                reason_display = gr.Markdown(elem_classes="why-panel")
                provider_note = gr.Markdown(elem_classes="provider-note")

        qa_group = gr.Column(visible=False)
        with qa_group:
            gr.Markdown("---", elem_classes="result-divider")
            with gr.Column(elem_classes="chat-shell"):
                with gr.Row(elem_classes="chat-header"):
                    with gr.Column(scale=4, min_width=280):
                        gr.Markdown(
                            "### Ask about this pun\n\n"
                            "Ask the selected model a follow-up question about the pun or analysis.",
                            elem_classes="chat-heading",
                            container=False,
                        )
                    provider_toggle = gr.Radio(
                        choices=AVAILABLE_PROVIDERS,
                        value=AVAILABLE_PROVIDERS[0],
                        label=None,
                        show_label=False,
                        container=False,
                        visible=True,
                        elem_id="provider-toggle",
                        scale=0,
                        min_width=0,
                    )
                thread_display = gr.Chatbot(
                    value=[],
                    type="messages",
                    layout="bubble",
                    height=240,
                    container=False,
                    autoscroll=True,
                    show_label=False,
                    show_copy_button=False,
                    feedback_options=None,
                    allow_tags=False,
                    visible=False,
                    elem_id="pun-chatbot",
                )
                with gr.Row(elem_classes="followup-row"):
                    followup_buttons = [
                        gr.Button(question, size="sm", variant="secondary", elem_classes="followup-chip")
                        for question in FOLLOW_UPS
                    ]
                with gr.Row(elem_id="question-row"):
                    msg_input = gr.Textbox(
                        label=None,
                        placeholder="Ask about this pun…",
                        lines=1, scale=5, show_label=False, container=False,
                        elem_id="question-input",
                    )
                    send_btn = gr.Button("Ask →", variant="primary", scale=1, elem_id="ask-button")

        bottom_group = gr.Column(visible=False)
        with bottom_group:
            gr.Markdown(
                "How it works: spaCy → WordNet → SBERT → Gemini/OpenAI\n\n"
                "SBERT · spaCy · WordNet · Gemini · OpenAI  ·  "
                "[View source on GitHub ↗](https://github.com/shria01/pun-dialog-interpreter)",
                elem_classes="bottom-meta",
            )

        chat_state = gr.State([])
        session_state = gr.State({"sentence": "", "analysis": None})

        analysis_outputs = [
            word_display, status_display, sense_a_display, sense_b_display,
            reason_display, provider_note, error_display, chat_state,
            thread_display, session_state, result_group, analysis_group,
            qa_group, bottom_group, analyze_btn, analysis_progress,
        ]

        for button, sentence in zip(example_buttons, EXAMPLE_PUNS.values()):
            button.click(
                lambda value=sentence: value,
                outputs=pun_input,
                queue=False,
            ).then(
                fn=analyze_with_progress,
                inputs=[pun_input, provider_toggle],
                outputs=analysis_outputs,
                show_progress="hidden",
            )

        for button, question in zip(followup_buttons, FOLLOW_UPS):
            button.click(
                fn=suggestion_handler(question),
                inputs=[chat_state, provider_toggle, session_state],
                outputs=[thread_display, chat_state, send_btn, msg_input],
                show_progress="hidden",
            )

        analyze_btn.click(
            fn=analyze_with_progress,
            inputs=[pun_input, provider_toggle],
            outputs=analysis_outputs,
            show_progress="hidden",
        )

        pun_input.submit(
            fn=analyze_with_progress,
            inputs=[pun_input, provider_toggle],
            outputs=analysis_outputs,
            show_progress="hidden",
        )

        send_btn.click(
            fn=respond_with_progress,
            inputs=[msg_input, chat_state, provider_toggle, session_state],
            outputs=[thread_display, chat_state, send_btn, msg_input],
            show_progress="hidden",
        )

        msg_input.submit(
            fn=respond_with_progress,
            inputs=[msg_input, chat_state, provider_toggle, session_state],
            outputs=[thread_display, chat_state, send_btn, msg_input],
            show_progress="hidden",
        )


demo.queue()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "8080")),
    )
