import nltk
import math
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
SENSE_MIN_CONTEXT_WEIGHT = 0.22
SENSE_MAX_CONTEXT_WEIGHT = 0.57
SENSE_DISTANCE_WEIGHT = 0.15
SENSE_FREQUENCY_WEIGHT = 0.10
MAX_SENSE_DISTANCE_REWARD = 0.50
PUNCHLINE_POSITION_WEIGHT = 0.26
INFLECTED_SENSE_WEIGHT = 0.15
MAX_INFLECTED_DISTANCE_REWARD = 0.70
CROSS_POS_SENSE_WEIGHT = 0.12
POS_AMBIGUITY_CANDIDATE_WEIGHT = 0.12


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


def score_definition_pair(
    fit_a,
    fit_b,
    sense_distance,
    frequency_score,
    surface_sense_match=False,
    cross_pos_match=False,
):
    """Rerank definitions without changing the pun-word ranking score.

    A pun commonly has one explicit meaning and one implied meaning, so this
    stage rewards the stronger contextual match more than the weaker one. The
    semantic-distance reward is capped to prevent unrelated definitions from
    winning simply because they are far apart.
    """
    return (
        SENSE_MIN_CONTEXT_WEIGHT * min(fit_a, fit_b)
        + SENSE_MAX_CONTEXT_WEIGHT * max(fit_a, fit_b)
        + SENSE_DISTANCE_WEIGHT * min(
            sense_distance,
            MAX_INFLECTED_DISTANCE_REWARD
            if surface_sense_match
            else MAX_SENSE_DISTANCE_REWARD,
        )
        + SENSE_FREQUENCY_WEIGHT * frequency_score
        + INFLECTED_SENSE_WEIGHT * surface_sense_match
        + CROSS_POS_SENSE_WEIGHT * cross_pos_match
    )


def synset_frequency_scores(synsets):
    """Return normalized WordNet usage priors for a word's candidate senses."""
    raw_scores = [
        math.log1p(sum(lemma.count() for lemma in synset.lemmas()))
        for synset in synsets
    ]
    maximum = max(raw_scores, default=0.0)
    if maximum == 0:
        return [0.0 for _ in raw_scores]
    return [score / maximum for score in raw_scores]


def select_definition_pair(
    synsets,
    embeddings,
    context_scores,
    surface_form=None,
):
    """Choose displayed senses after the word itself has been scored."""
    frequency_scores = synset_frequency_scores(synsets)
    surface = (surface_form or "").lower()
    base_form = wn.morphy(surface, wn.VERB) if surface else None
    is_inflected = bool(base_form and base_form != surface)
    pairs = []
    for first, second in combinations(range(len(synsets)), 2):
        distance = 1 - util.cos_sim(
            embeddings[first],
            embeddings[second],
        ).item()
        frequency_score = (
            frequency_scores[first] + frequency_scores[second]
        ) / 2
        surface_sense_match = is_inflected and any(
            lemma.name().lower() == surface
            for index in (first, second)
            for lemma in synsets[index].lemmas()
        )
        cross_pos_match = synsets[first].pos() != synsets[second].pos()
        pairs.append({
            "score": score_definition_pair(
                context_scores[first],
                context_scores[second],
                distance,
                frequency_score,
                surface_sense_match,
                cross_pos_match,
            ),
            "first": first,
            "second": second,
            "fit_a": context_scores[first],
            "fit_b": context_scores[second],
            "distance": distance,
            "frequency_score": frequency_score,
            "cross_pos_match": cross_pos_match,
        })

    distinct_pairs = [
        pair for pair in pairs
        if pair["distance"] >= MIN_SENSE_DISTANCE
    ]
    return max(distinct_pairs or pairs, key=lambda pair: pair["score"])


def parse_sentence(sentence: str):
    """Tokenize and tag the sentence with spaCy."""
    return nlp(sentence)


def retrieve_wordnet_candidates(doc):
    """Collect WordNet senses for eligible single-word tokens."""
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
        if token.tag_ in {"VBN", "VBG"}:
            participle_synsets = (
                wn.synsets(token.lemma_, pos=wn.VERB)
                + wn.synsets(token.text, pos=wn.ADJ)
            )
            synsets = list(dict.fromkeys([*synsets, *participle_synsets]))
        if len(synsets) < 2:
            continue

        descriptions = [sense_text(synset) for synset in synsets]
        candidates.append((token.text, synsets, descriptions))

    # If the same written word appears with different parts of speech, combine
    # those inventories into one candidate. This captures single-word
    # syntactic puns such as verb "flies" versus noun "flies" without adding
    # support for compound or phrase-based pun targets.
    grouped_candidates = {}
    candidate_order = []
    for surface, synsets, _ in candidates:
        key = surface.lower()
        if key not in grouped_candidates:
            grouped_candidates[key] = [surface, []]
            candidate_order.append(key)
        grouped_candidates[key][1].extend(synsets)

    candidates = []
    for key in candidate_order:
        surface, synsets = grouped_candidates[key]
        synsets = list(dict.fromkeys(synsets))
        descriptions = [sense_text(synset) for synset in synsets]
        candidates.append((surface, synsets, descriptions))

    if not candidates:
        raise ValueError("No candidate pun words found in sentence.")
    return candidates


def rank_sense_candidates(sentence: str, candidates) -> list[dict]:
    """Rank words by their strongest contextually valid sense pair."""
    started = time.perf_counter()
    word_scores = []
    doc = nlp(sentence)
    parts_of_speech = {}
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
            parts_of_speech.setdefault(token.lemma_.lower(), set()).add(token.pos_)
    sentence_pos_ambiguities = [
        {
            "lemma": lemma,
            "parts_of_speech": sorted(pos_values),
        }
        for lemma, pos_values in parts_of_speech.items()
        if len(pos_values) > 1
    ]

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

        # Keep best_pair and its score as the committed word-ranking signal.
        # Select the two definitions separately so improving their quality cannot
        # change which word wins.
        definition_pair = select_definition_pair(
            synsets,
            embeddings,
            context_scores,
            surface_form=word,
        )
        first = definition_pair["first"]
        second = definition_pair["second"]
        if definition_pair["fit_b"] > definition_pair["fit_a"]:
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
            "definition_pair_score": definition_pair["score"],
            "definition_context_fit": min(
                definition_pair["fit_a"],
                definition_pair["fit_b"],
            ),
            "definition_sense_distance": definition_pair["distance"],
            "definition_frequency_score": definition_pair["frequency_score"],
            "definition_cross_pos_match": definition_pair["cross_pos_match"],
            "sentence_pos_ambiguities": sentence_pos_ambiguities,
        })

    # Preserve the original score as the candidate generator, then calibrate a
    # small shortlist. Puns commonly place the payoff after the clue, so later
    # words receive a modest position prior. Definition-pair quality prevents a
    # highly polysemous clue noun from winning on distance alone.
    ranked_by_word_score = sorted(
        word_scores,
        key=lambda candidate: candidate["pun_score"],
        reverse=True,
    )
    shortlist = ranked_by_word_score[:5]
    denominator = max(len(doc) - 1, 1)
    ambiguous_lemmas = {
        ambiguity["lemma"] for ambiguity in sentence_pos_ambiguities
    }
    positions = {}
    for token in doc:
        position = token.i / denominator
        for form in {token.lemma_.lower(), token.text.lower()}:
            positions[form] = max(positions.get(form, 0.0), position)
    for candidate in shortlist:
        candidate["position_score"] = positions.get(
            candidate["word"].lower(),
            positions.get(candidate["word"].split()[-1].lower(), 0.0),
        )
        candidate["selection_score"] = (
            candidate["definition_pair_score"]
            + PUNCHLINE_POSITION_WEIGHT * candidate["position_score"]
            + POS_AMBIGUITY_CANDIDATE_WEIGHT
            * any(
                token.text.lower() == candidate["word"].lower()
                and token.lemma_.lower() in ambiguous_lemmas
                for token in doc
            )
        )
    top_candidates = sorted(
        shortlist,
        key=lambda candidate: candidate["selection_score"],
        reverse=True,
    )[:3]
    print(
        f"timing rank_senses={time.perf_counter() - started:.3f}s "
        f"cache={len(_embedding_cache)}",
        flush=True,
    )
    return top_candidates
