import nltk
import spacy
import threading
import time
import torch
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn


CONTEXT_WEIGHT = 0.65
DISTANCE_WEIGHT = 0.35
BALANCE_PENALTY = 0.15
MIN_SENSE_DISTANCE = 0.15


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
    """Encode unseen sense descriptions in one batch and reuse them."""
    unique_definitions = list(dict.fromkeys(definitions))
    with _embedding_lock:
        missing = [
            definition
            for definition in unique_definitions
            if definition not in _embedding_cache
        ]
        if missing:
            embeddings = model.encode(missing, convert_to_tensor=True)
            for definition, embedding in zip(missing, embeddings):
                _embedding_cache[definition] = embedding.cpu()
        return {
            definition: _embedding_cache[definition]
            for definition in unique_definitions
        }


def sense_text(synset):
    """Build a sentence-like sense description for embedding."""
    lemmas = ", ".join(
        lemma.name().replace("_", " ")
        for lemma in synset.lemmas()
    )
    examples = " ".join(synset.examples())
    parts = [f"Meaning: {synset.definition()}."]
    if lemmas:
        parts.append(f"Related words: {lemmas}.")
    if examples:
        parts.append(f"Examples: {examples}")
    return " ".join(parts)


def score_sense_pair(fit_a, fit_b, sense_distance):
    """Reward two contextually plausible but semantically distinct senses."""
    context_fit = min(fit_a, fit_b)
    balance = abs(fit_a - fit_b)
    return (
        CONTEXT_WEIGHT * context_fit
        + DISTANCE_WEIGHT * sense_distance
        - BALANCE_PENALTY * balance
    )


def parse_sentence(sentence: str):
    """Tokenize and tag the sentence with spaCy."""
    return nlp(sentence)


def retrieve_wordnet_candidates(doc):
    """Collect WordNet senses for unique eligible spaCy tokens."""
    candidates = []
    seen = set()
    pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ}

    for token in doc:
        if token.pos_ not in pos_map:
            continue

        candidate_key = (token.lemma_.lower(), token.pos_)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)

        wn_pos = pos_map[token.pos_]
        synsets = wn.synsets(token.lemma_, pos=wn_pos) or wn.synsets(token.lemma_)
        if len(synsets) < 2:
            continue

        descriptions = [sense_text(synset) for synset in synsets]
        candidates.append((token.lemma_, synsets, descriptions))

    if not candidates:
        raise ValueError("No candidate pun words found in sentence.")
    return candidates


def rank_sense_candidates(sentence: str, candidates) -> list[dict]:
    """Rank words by their strongest contextually valid sense pair."""
    started = time.perf_counter()
    word_scores = []

    sense_embeddings = encode_definitions([
        description
        for _, _, descriptions in candidates
        for description in descriptions
    ])
    with _embedding_lock:
        sentence_embedding = model.encode(
            sentence,
            convert_to_tensor=True,
        ).cpu()

    for word, synsets, descriptions in candidates:
        embeddings = [sense_embeddings[text] for text in descriptions]
        context_scores = util.cos_sim(
            sentence_embedding,
            torch.stack(embeddings),
        )[0].tolist()

        pairs = []
        for first, second in combinations(range(len(synsets)), 2):
            distance = 1 - util.cos_sim(
                embeddings[first],
                embeddings[second],
            ).item()
            pair = {
                "score": score_sense_pair(
                    context_scores[first],
                    context_scores[second],
                    distance,
                ),
                "first": first,
                "second": second,
                "fit_a": context_scores[first],
                "fit_b": context_scores[second],
                "distance": distance,
            }
            pairs.append(pair)

        distinct_pairs = [
            pair for pair in pairs
            if pair["distance"] >= MIN_SENSE_DISTANCE
        ]
        best_pair = max(distinct_pairs or pairs, key=lambda pair: pair["score"])

        first = best_pair["first"]
        second = best_pair["second"]
        if best_pair["fit_b"] > best_pair["fit_a"]:
            first, second = second, first

        word_scores.append({
            "word": word,
            "pun_score": best_pair["score"],
            "sense_a": synsets[first].definition(),
            "sense_b": synsets[second].definition(),
            "sense_a_id": synsets[first].name(),
            "sense_b_id": synsets[second].name(),
            "context_fit": min(best_pair["fit_a"], best_pair["fit_b"]),
            "sense_distance": best_pair["distance"],
            "fit_balance": abs(best_pair["fit_a"] - best_pair["fit_b"]),
        })

    top_candidates = sorted(
        word_scores,
        key=lambda candidate: candidate["pun_score"],
        reverse=True,
    )[:3]
    print(
        f"timing rank_senses={time.perf_counter() - started:.3f}s "
        f"cache={len(_embedding_cache)}",
        flush=True,
    )
    return top_candidates
