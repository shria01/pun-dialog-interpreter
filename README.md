# Pun Dialog Interpreter

A conversational AI system that identifies pun words, explains the humor, and answers follow-up questions. Built with Python, SBERT, spaCy, WordNet, Gradio, and your choice of Google Gemini or OpenAI GPT.

**[Try the live app →](https://pun-dialog-interpreter-234564595283.us-west1.run.app)**


## How It Works

1. Enter a pun or pick one of the examples
2. Click Analyze — the system identifies the pun word and its two meanings
3. Ask questions in the chat about why it's funny
4. Switch between Gemini and OpenAI using the model toggle in the chat header


## Architecture

When a user enters a pun, `sense_finder` tokenizes it with spaCy and looks up WordNet synsets for each noun, verb, and adjective. It enriches each sense with its definition, lemmas, and usage examples, then uses SBERT to compare those descriptions with the sentence. Every possible sense pair is scored by contextual fit, semantic distance, and balance. The highest-scoring pair determines each word's score, and the highest-scoring word becomes the final detected pun word. `context_validator` asks the selected LLM to validate and explain that fixed result; the LLM cannot replace the word or rewrite its WordNet senses.

The LLM layer uses an abstract interface built with Python's ABC module. Gemini and OpenAI both implement the same `generate` and `chat` methods. Gemini converts the standard message format to its own format internally and OpenAI passes it through as is. Adding another provider just means implementing those two methods.

## Examples to try

- "I used to be a banker but I lost interest"
- "The math teacher was a good ruler"
- "A boiled egg in the morning is hard to beat"
- "I used to hate facial hair but then it grew on me"
- "Broken pencils are pointless"

## Pun scoring

For every possible pair of senses belonging to a content word, `sense_finder` computes:

```
pair_score =
    0.65 × min(sense_a_similarity, sense_b_similarity)
  + 0.35 × semantic_distance
  - 0.15 × abs(sense_a_similarity - sense_b_similarity)
```

The minimum similarity requires both senses to fit the context, semantic distance rewards a meaningful contrast, and the final term penalizes pairs where only one sense fits well. Pairs with near-duplicate meanings are filtered out. The strongest pair determines the word's score, and the strongest word is selected without LLM intervention.

Run the detector-only benchmark without using either LLM API:

```bash
python scripts/evaluate_detector.py
```

**Known limitation:** WordNet indexes individual words, so phrasal verbs and idioms like "put down" or "give up" are partially missed — the idiomatic meaning doesn't exist on either word alone.

## My contributions

Co-authored the original pun scoring and WordNet-based detection as part of a Purdue group project, then in this fork extended it. In the original project, I solely built the sense_finder module — SBERT embeddings, POS-aware WordNet lookup, and the pun scoring formula. (context_validator and dialog_bot were built by teammates.)

In this fork, I extended the system by:

- Reworking detection to rank every possible WordNet sense pair using contextual fit, semantic distance, and fit balance
- Enriching sense embeddings with WordNet lemmas and usage examples instead of embedding dictionary definitions alone
- Making the SBERT ranking result final while restricting the LLM to validation and explanation
- Refactoring the LLM layer into an abstract provider interface so Gemini and OpenAI are swappable without touching the rest of the code
- Adding OpenAI support with runtime provider switching in the UI
- Adding retry logic and better JSON parsing for structured outputs

## Author

Shria Kondragunta — [github.com/shria01](https://github.com/shria01)

Original project: [ssuwaneh/Dialog-Pun-Interpreter-Group-10-NLP-Project](https://github.com/ssuwaneh/Dialog-Pun-Interpreter-Group-10-NLP-Project)
