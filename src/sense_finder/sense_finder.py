"""
LIMITATION: WordNet indexes senses by individual words, so phrasal verbs 
and idioms (e.g. "put down", "give up") are missed entirely. The pun in
"impossible to put down" relies on the idiomatic sense of "put down" 
(to stop reading) vs. the literal sense (to place downward), but since
"put" and "down" are tokenized separately, neither word alone carries 
the idiomatic meaning.

IMPROVEMENT: Returns top 3 candidate pun words instead of just the best,
allowing the LLM to make the final selection based on linguistic context.
"""

import nltk
import spacy
import threading
import time
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn


def ensure_nltk_data():
  """Install WordNet data on first startup in a fresh hosted environment."""
  resources = {
      "corpora/wordnet": "wordnet",
      "corpora/omw-1.4": "omw-1.4",
  }
  for resource_path, package_name in resources.items():
    try:
      nltk.data.find(resource_path)
    except LookupError:
      nltk.download(package_name, quiet=True)


ensure_nltk_data()
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")
_embedding_cache = {}
_embedding_lock = threading.Lock()


def encode_definitions(definitions):
  """Encode unseen definitions in one batch and reuse them across requests."""
  unique_definitions = list(dict.fromkeys(definitions))
  with _embedding_lock:
    missing = [definition for definition in unique_definitions if definition not in _embedding_cache]
    if missing:
      embeddings = model.encode(missing, convert_to_tensor=True)
      for definition, embedding in zip(missing, embeddings):
        _embedding_cache[definition] = embedding.cpu()
    return {definition: _embedding_cache[definition] for definition in unique_definitions}

def parse_sentence(sentence: str):
  """Tokenize and tag the sentence with spaCy."""
  return nlp(sentence)


def retrieve_wordnet_candidates(doc):
  """Collect WordNet senses for eligible spaCy tokens."""
  candidates = []

  for token in doc:
    if token.pos_ not in ("NOUN", "VERB", "ADJ"):
      continue
    pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ}
    wn_pos = pos_map.get(token.pos_)
    synsets = wn.synsets(token.lemma_, pos=wn_pos) or wn.synsets(token.lemma_)

    if len(synsets) < 2:
      continue

    definitions = [s.definition() for s in synsets]
    candidates.append((token.lemma_, synsets, definitions))

  if not candidates:
    raise ValueError("No candidate pun words found in sentence.")
  return candidates


def rank_sense_candidates(sentence: str, candidates) -> list[dict]:
  """Use SBERT similarity to rank the candidate words and senses."""
  started = time.perf_counter()
  scores = []

  definition_embeddings = encode_definitions([
      definition
      for _, _, definitions in candidates
      for definition in definitions
  ])
  with _embedding_lock:
    sent_emb = model.encode(sentence, convert_to_tensor=True).cpu()

  for word, synsets, definitions in candidates:
    def_embs = [definition_embeddings[definition] for definition in definitions]
    stacked_def_embs = torch.stack(def_embs)
    sense_scores = util.cos_sim(sent_emb, stacked_def_embs)[0].tolist()
    
    ranked = sorted(zip(sense_scores, definitions, synsets), reverse=True)
    top1_score = ranked[0][0]
    top2_score = ranked[1][0] if len(ranked) > 1 else 0
    
    top1_emb = definition_embeddings[ranked[0][1]]
    top2_emb = definition_embeddings[ranked[1][1]]
    
    sense_distance = 1 - util.cos_sim(top1_emb, top2_emb).item()
    pun_score = (top1_score + top2_score) / 2 * sense_distance
    
    scores.append({
        "word": word,
        "pun_score": pun_score,
        "sense_a": ranked[0][1],
        "sense_b": ranked[1][1],
        })

  # Return top 3 instead of just best.
  top3 = sorted(scores, key=lambda x: x["pun_score"], reverse=True)[:3]
  print(f"timing rank_senses={time.perf_counter() - started:.3f}s cache={len(_embedding_cache)}", flush=True)
  return top3
