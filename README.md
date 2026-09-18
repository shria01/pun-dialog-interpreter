# Pun Detector

An NLP system for single-word puns that detects the word carrying the wordplay, ranks its competing meanings, and shows why the double meaning works. Compound and phrase-based pun targets are outside the detector's current scope. The detector is built with spaCy, WordNet, and SBERT; Gemini or OpenAI validates the fixed detection and provides optional follow-up explanations.

**[Try the live app →](https://pun-dialog-interpreter-234564595283.us-west1.run.app)**


## How It Works

1. Enter a sentence or choose an example.
2. The detector extracts content words with spaCy and retrieves their possible WordNet senses.
3. SBERT ranks every viable sense pair and selects the most likely pun word without LLM intervention.
4. Gemini or OpenAI validates the fixed result and explains confirmed detections, ranking failures, or sentences that are not puns.
5. For confirmed puns or detector failures, ask optional follow-up questions about the analysis.


## Architecture

When a user enters a sentence, `sense_finder` tokenizes it with spaCy and looks up WordNet synsets for each noun, verb, and adjective. It enriches each sense with its definition, lemmas, and usage examples, then uses SBERT to compare those descriptions with the sentence. Every possible sense pair is scored by contextual fit, semantic distance, and balance. That original score generates a three-word shortlist. A second stage reranks the shortlist using definition-pair quality and a modest punchline-position prior, then returns definitions selected with contextual fit, capped semantic distance, and WordNet usage frequency.

Detection remains owned by the NLP pipeline. `context_validator` can classify the fixed result as a confirmed pun, a wrong detected word, a correct word paired with the wrong senses, or not a pun, but it cannot replace the selected word or rewrite its WordNet senses. The UI presents these outcomes differently while leaving the committed scoring model unchanged.

The secondary explanation layer uses an abstract interface built with Python's ABC module. Gemini and OpenAI implement the same `generate` and `chat` methods, so the provider can be switched without changing the detector.

## Examples to try

- "I used to be a banker but I lost interest"
- "The math teacher was a good ruler"
- "A boiled egg in the morning is hard to beat"
- "Broken pencils are pointless"

## Detection scoring

For every possible pair of senses belonging to a content word, `sense_finder` computes:

```
pair_score =
    0.65 × min(sense_a_similarity, sense_b_similarity)
  + 0.35 × semantic_distance
  - 0.15 × abs(sense_a_similarity - sense_b_similarity)
```

The minimum similarity requires both senses to fit the context, semantic distance rewards a meaningful contrast, and the final term penalizes pairs where only one sense fits well. Pairs with near-duplicate meanings are filtered out. The strongest pair determines the word's score, and the strongest word is selected without LLM intervention.

After word scores are fixed, definition selection uses a separate score:

```text
definition_score =
    0.22 × min(context similarities)
  + 0.57 × max(context similarities)
  + 0.15 × min(semantic distance, 0.50)
  + 0.10 × normalized WordNet frequency
```

This gives more weight to the explicit meaning, allows the second pun meaning to be implicit, caps the reward for unrelated senses, and uses WordNet frequency only as a light prior.

The original word score still generates the top-three candidate set. Final selection within that shortlist is:

```text
selection_score =
    definition_score
  + 0.26 × normalized_token_position
```

The position prior reflects the common clue-then-payoff structure of short puns and helps prevent a related clue noun such as “pencil” from outranking the actual double-meaning word “pointless.”

Inflected participles such as “hooked” are retrieved across both verb and adjective WordNet senses. The detector preserves the surface form for display and gives a small preference to senses explicitly indexed under that inflected form, while still requiring contextual fit and semantic contrast.

The detector intentionally ranks single-word candidates. Puns whose double meaning depends on a compound or phrasal expression are reported as outside the current detector scope.

When the same written word appears under different parts of speech, the detector merges those WordNet inventories and rewards a cross-POS sense pair. This supports single-word structural ambiguity such as verb “flies” versus noun “flies” without treating a multiword phrase as the pun target.

Run the detector-only regression suite without using either LLM API. It checks
the documented candidate and sense-selection examples; it is not a general
accuracy estimate or an end-to-end evaluation of LLM validation:

```bash
python scripts/evaluate_detector.py
```

**Known limitation:** The detector only supports puns whose competing meanings are carried by one written word. Compound expressions, phrasal verbs, and idioms such as "put down" or "give up" are outside its current scope.

## My contributions

Co-authored the original pun scoring and WordNet-based detection as part of a Purdue group project, then in this fork extended it. In the original project, I solely built the sense_finder module — SBERT embeddings, POS-aware WordNet lookup, and the pun scoring formula. (context_validator and dialog_bot were built by teammates.)

In this fork, I extended the system by:

- Reworking detection to rank every possible WordNet sense pair using contextual fit, semantic distance, and fit balance
- Enriching sense embeddings with WordNet lemmas and usage examples instead of embedding dictionary definitions alone
- Making the SBERT ranking result final while restricting the LLM to validation and explanation
- Distinguishing confirmed detections, ranking failures, and non-pun inputs in the result UI
- Separating wrong-word failures from correct-word/wrong-sense failures
- Refactoring the LLM layer into an abstract provider interface so Gemini and OpenAI are swappable without touching the rest of the code
- Adding OpenAI support with runtime provider switching in the UI
- Adding retry logic and better JSON parsing for structured outputs

## Author

Shria Kondragunta — [github.com/shria01](https://github.com/shria01)

Original project: [ssuwaneh/Dialog-Pun-Interpreter-Group-10-NLP-Project](https://github.com/ssuwaneh/Dialog-Pun-Interpreter-Group-10-NLP-Project)
