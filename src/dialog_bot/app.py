import html
import os
import sys

import gradio as gr

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(SRC_DIR)

from dialog_bot import analyze_pun, chat
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

APP_CSS = """
:root {
  --page: #0b0f17; --surface: #111827; --surface-2: #172033;
  --border: #263244; --text: #f3f4f6; --muted: #94a3b8;
  --accent: #6366f1; --positive: #34d399;
}
.gradio-container { background: var(--page) !important; color: var(--text) !important; }
#app-shell { max-width: 1100px; margin: 0 auto; padding: 32px 24px 48px; gap: 0; }
.hero { margin-bottom: 24px; }
.eyebrow, .section-label, .meaning-label {
  color: var(--muted); font-size: 12px; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
}
.hero h1 { color: var(--text); font-size: clamp(32px, 5vw, 44px); line-height: 1.05; margin: 8px 0 12px; }
.hero-copy { color: var(--text); font-size: 18px; margin: 0 0 6px; }
.hero-tech { color: var(--muted); font-size: 14px; margin: 0; }
#pun-input textarea, #question-input textarea {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text) !important;
  font-size: 16px !important; min-height: 52px !important;
  transition: border-color 150ms ease, box-shadow 150ms ease;
}
#pun-input textarea:focus, #question-input textarea:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgb(99 102 241 / 18%) !important;
}
#analyze-button, #ask-button {
  background: var(--accent) !important; border: 0 !important; border-radius: 8px !important;
  color: white !important; font-weight: 700 !important; min-height: 52px;
  transition: filter 150ms ease, transform 150ms ease;
}
#analyze-button:hover, #ask-button:hover { filter: brightness(1.1); transform: translateY(-1px); }
#provider-selector {
  background: var(--surface); border: 1px solid var(--border); border-radius: 9px;
  display: inline-flex; margin: 10px 0 24px; padding: 3px; width: fit-content;
}
#provider-selector > div { gap: 3px !important; }
#provider-selector label {
  background: transparent !important; border: 0 !important; border-radius: 6px !important;
  color: var(--muted) !important; min-width: 112px; padding: 8px 12px !important;
  transition: background 150ms ease, color 150ms ease;
}
#provider-selector input[type="radio"] { display: none !important; }
#provider-selector label:has(input:checked) {
  background: rgb(99 102 241 / 18%) !important; color: var(--text) !important;
  box-shadow: inset 0 0 0 1px rgb(99 102 241 / 45%);
}
#provider-selector label:has(input:checked)::after { color: #a5b4fc; content: "✓"; margin-left: 8px; }
.example-row, .followup-row { flex-wrap: wrap !important; gap: 8px; margin: 8px 0 32px; }
.example-chip, .followup-chip {
  background: transparent !important; border: 1px solid var(--border) !important;
  border-radius: 6px !important; color: var(--muted) !important;
  font-size: 13px !important; min-width: auto !important;
  transition: color 150ms ease, border-color 150ms ease, background 150ms ease;
}
.example-chip:hover, .followup-chip:hover { background: var(--surface) !important; border-color: #475569 !important; color: var(--text) !important; }
.section-divider { border-top: 1px solid var(--border); margin: 16px 0 24px; padding-top: 24px; }
#analysis-section .section-divider { margin-top: 24px; padding-top: 32px; }
.analysis-result { margin-bottom: 40px; }
.result-heading { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin: 10px 0 20px; }
.pun-word { color: var(--text); font-size: clamp(38px, 6vw, 48px); font-weight: 800; letter-spacing: -.01em; }
.status { color: var(--positive); font-size: 14px; font-weight: 700; white-space: nowrap; }
.status.neutral { color: var(--muted); }
.meaning-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.meaning-card { background: #151e2e; border: 1px solid #334155; border-radius: 12px; min-height: 150px; padding: 20px; }
.meaning-text { color: var(--text); font-size: 16px; line-height: 1.6; margin: 18px 0 0; }
.meaning-bridge { color: var(--muted); font-size: 13px; text-align: center; margin: 14px 0 28px; }
.why-panel { background: rgb(99 102 241 / 7%); border-left: 2px solid var(--accent); border-radius: 0 8px 8px 0; padding: 16px 18px; }
.why-title { color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.why-copy { color: var(--text); font-size: 16px; line-height: 1.65; margin-bottom: 0; max-width: 780px; }
.provider-note { color: #64748b; font-size: 11px; margin-top: 12px; }
.error-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.error-card h3 { color: var(--text); margin: 0 0 8px; }
.error-card p { color: var(--muted); margin: 0; }
#chatbot { background: transparent !important; border: 0 !important; max-height: 340px !important; min-height: 0 !important; overflow-y: auto !important; }
#chatbot .message { border-radius: 10px !important; padding: 12px 14px !important; }
.how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 18px 0 36px; }
.how-step { border-top: 1px solid var(--border); padding-top: 14px; }
.how-step strong { color: var(--text); display: block; margin: 6px 0; }
.how-step span:last-child { color: var(--muted); font-size: 13px; line-height: 1.5; }
.step-number { color: var(--accent); font-size: 12px; font-weight: 700; }
.footer { border-top: 1px solid var(--border); color: var(--muted); display: flex; justify-content: space-between; padding-top: 20px; font-size: 13px; }
.footer-meta { display: flex; flex-direction: column; gap: 3px; }
.footer-project { color: #64748b; font-size: 11px; }
.footer a { color: #a5b4fc; text-decoration: none; }
@media (max-width: 700px) {
  #app-shell { padding: 24px 16px 36px; }
  .meaning-grid, .how-grid { grid-template-columns: 1fr; }
  .result-heading, .footer { align-items: flex-start; flex-direction: column; }
  #input-row, #question-row { flex-direction: column; }
  #analyze-button, #ask-button { width: 100%; }
}
"""


def get_provider(name):
    if name == "Gemini":
        return GeminiProvider()
    if name == "OpenAI":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")


def to_pairs(history):
    """Convert Gradio message dictionaries into user/assistant pairs."""
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def render_analysis(analysis, provider_name):
    """Render model output as a safe, structured analysis panel."""
    word = html.escape(str(analysis.get("pun_word", "No clear pun")))
    sense_a = html.escape(str(analysis.get("sense_a", "No first meaning found.")))
    sense_b = html.escape(str(analysis.get("sense_b", "No second meaning found.")))
    reason = html.escape(str(analysis.get("reason", "No explanation was returned.")))
    provider = html.escape(provider_name)
    works = bool(analysis.get("pun_works"))
    status = "✓ Double meaning confirmed" if works else "No clear pun detected"
    status_class = "status" if works else "status neutral"
    return f"""
    <div class="analysis-result">
      <div class="section-label">Analysis</div>
      <div class="result-heading">
        <div class="pun-word">{word}</div>
        <div class="{status_class}">{status}</div>
      </div>
      <div class="meaning-grid">
        <article class="meaning-card"><div class="meaning-label">Meaning 01</div><p class="meaning-text">{sense_a}</p></article>
        <article class="meaning-card"><div class="meaning-label">Meaning 02</div><p class="meaning-text">{sense_b}</p></article>
      </div>
      <div class="meaning-bridge">Both meanings apply to the same word in context.</div>
      <div class="why-panel">
        <div class="why-title">Why it works</div>
        <p class="why-copy">{reason}</p>
      </div>
      <div class="provider-note">Analysis generated with {provider}</div>
    </div>
    """


def render_error(provider_name, error):
    provider = html.escape(provider_name)
    message = (
        "Try a sentence containing a word with two possible meanings."
        if isinstance(error, ValueError)
        else f"The {provider} request could not be completed. Try again or switch models."
    )
    return f'<div class="error-card"><h3>Couldn\'t analyze this sentence</h3><p>{message}</p></div>'


def start_analysis():
    return gr.update(value="Analyzing…", interactive=False)


def finish_analysis():
    return gr.update(value="Analyze Pun →", interactive=True)


def set_pun(sentence, provider_name):
    """Analyze a pun and reveal the result and follow-up controls."""
    sentence = sentence.strip()
    if not sentence:
        return (render_error(provider_name, ValueError("Empty sentence")), [],
                gr.update(value=[], visible=False), {"sentence": "", "analysis": None},
                gr.update(visible=True), gr.update(visible=False))
    try:
        analysis = analyze_pun(sentence, get_provider(provider_name))
    except Exception as error:
        return (render_error(provider_name, error), [], gr.update(value=[], visible=False),
                {"sentence": "", "analysis": None}, gr.update(visible=True),
                gr.update(visible=False))

    session = {"sentence": sentence, "analysis": analysis}
    return (render_analysis(analysis, provider_name), [], gr.update(value=[], visible=False),
            session, gr.update(visible=True), gr.update(visible=True))


def respond(question, history, provider_name, session):
    """Generate a follow-up response and reveal the compact conversation."""
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
    return gr.update(value=history, visible=True), history


with gr.Blocks(
    title="Pun Interpreter",
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    css=APP_CSS,
) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML("""
        <header class="hero">
          <div class="eyebrow">NLP · Word sense disambiguation</div>
          <h1>Pun Interpreter</h1>
          <p class="hero-copy">Decode the double meaning behind a pun.</p>
          <p class="hero-tech">Semantic analysis using WordNet, SBERT, spaCy, and LLM reasoning.</p>
        </header>
        """)
        gr.HTML('<div class="section-label">Try a pun</div>')
        with gr.Row(elem_id="input-row"):
            pun_input = gr.Textbox(placeholder="Broken pencils are pointless", show_label=False,
                                   lines=1, scale=5, elem_id="pun-input")
            analyze_btn = gr.Button("Analyze Pun →", variant="primary", scale=1,
                                    elem_id="analyze-button")

        gr.HTML('<div class="section-label" style="margin-top:16px">Choose AI</div>')
        provider_toggle = gr.Radio(choices=AVAILABLE_PROVIDERS, value=AVAILABLE_PROVIDERS[0],
                                   show_label=False, container=False, elem_id="provider-selector")

        gr.HTML('<div class="section-label">Try an example</div>')
        with gr.Row(elem_classes="example-row"):
            example_buttons = [gr.Button(label, size="sm", elem_classes="example-chip")
                               for label in EXAMPLE_PUNS]

        analysis_group = gr.Column(visible=False, elem_id="analysis-section")
        with analysis_group:
            gr.HTML('<div class="section-divider"></div>')
            analysis_display = gr.HTML()

        chat_group = gr.Column(visible=False)
        with chat_group:
            gr.HTML('<div class="section-divider"><div class="section-label">Ask about this pun</div></div>')
            with gr.Row(elem_id="question-row"):
                msg_input = gr.Textbox(placeholder="Why is the second meaning funny?",
                                       show_label=False, lines=1, scale=5, elem_id="question-input")
                send_btn = gr.Button("Ask →", variant="primary", scale=1, elem_id="ask-button")
            gr.HTML('<div class="section-label" style="margin-top:12px">Suggested follow-ups</div>')
            with gr.Row(elem_classes="followup-row"):
                followup_buttons = [gr.Button(text, size="sm", elem_classes="followup-chip")
                                    for text in FOLLOW_UPS]
            chatbot = gr.Chatbot(show_label=False, type="messages", visible=False,
                                 height=340, allow_tags=False, elem_id="chatbot")

        gr.HTML("""
        <section class="section-divider">
          <div class="section-label">How it works</div>
          <div class="how-grid">
            <div class="how-step"><span class="step-number">01</span><strong>Detect</strong><span>Identify the likely pun word.</span></div>
            <div class="how-step"><span class="step-number">02</span><strong>Interpret</strong><span>Retrieve competing lexical meanings.</span></div>
            <div class="how-step"><span class="step-number">03</span><strong>Validate</strong><span>Check both meanings against context.</span></div>
            <div class="how-step"><span class="step-number">04</span><strong>Explain</strong><span>Generate a natural-language explanation.</span></div>
          </div>
          <footer class="footer">
            <span class="footer-meta">
              <span>SBERT · spaCy · WordNet · Gemini · OpenAI</span>
              <span class="footer-project">Portfolio project · NLP / ML</span>
            </span>
            <a href="https://github.com/shria01/pun-dialog-interpreter" target="_blank">View source on GitHub ↗</a>
          </footer>
        </section>
        """)

        chat_state = gr.State([])
        session_state = gr.State({"sentence": "", "analysis": None})

        for button, sentence in zip(example_buttons, EXAMPLE_PUNS.values()):
            button.click(lambda value=sentence: value, outputs=pun_input, queue=False)
        for button, question in zip(followup_buttons, FOLLOW_UPS):
            button.click(lambda value=question: value, outputs=msg_input, queue=False)

        analysis_event = analyze_btn.click(fn=start_analysis, outputs=analyze_btn, queue=False).then(
            fn=set_pun,
            inputs=[pun_input, provider_toggle],
            outputs=[analysis_display, chat_state, chatbot, session_state,
                     analysis_group, chat_group],
        )
        analysis_event.then(fn=finish_analysis, outputs=analyze_btn, queue=False)

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chat_state, provider_toggle, session_state],
            outputs=[chatbot, chat_state],
        ).then(lambda: "", outputs=msg_input, queue=False)
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chat_state, provider_toggle, session_state],
            outputs=[chatbot, chat_state],
        ).then(lambda: "", outputs=msg_input, queue=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "8080")))
