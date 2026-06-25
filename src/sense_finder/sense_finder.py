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
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

def find_senses(sentence: str) -> list[dict]:
  doc = nlp(sentence)
  scores = []
  sent_emb = model.encode(sentence)

  for token in doc:
    if token.pos_ not in ("NOUN", "VERB", "ADJ"):
      continue
    pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ}
    wn_pos = pos_map.get(token.pos_)
    synsets = wn.synsets(token.lemma_, pos=wn_pos) or wn.synsets(token.lemma_)

    if len(synsets) < 2:
      continue

    definitions = [s.definition() for s in synsets]
    def_embs = model.encode(definitions)
    sense_scores = util.cos_sim(sent_emb, def_embs)[0].tolist()
    
    ranked = sorted(zip(sense_scores, definitions, synsets), reverse=True)
    top1_score = ranked[0][0]
    top2_score = ranked[1][0] if len(ranked) > 1 else 0
    
    top1_emb = def_embs[definitions.index(ranked[0][1])]
    top2_emb = def_embs[definitions.index(ranked[1][1])]
    
    sense_distance = 1 - util.cos_sim(top1_emb, top2_emb).item()
    pun_score = (top1_score + top2_score) / 2 * sense_distance
    
    scores.append({
        "word": token.lemma_,
        "pun_score": pun_score,
        "sense_a": ranked[0][1],
        "sense_b": ranked[1][1],
        })

  if not scores:
        raise ValueError("No candidate pun words found in sentence.")

    # return top 3 instead of just best
  top3 = sorted(scores, key=lambda x: x["pun_score"], reverse=True)[:3]
  return top3
