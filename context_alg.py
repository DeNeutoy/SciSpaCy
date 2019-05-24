
from typing import NamedTuple, Optional, List, Dict, Any, Set
import json
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher

class ContextPattern(NamedTuple):

    name: str
    label: str
    regex: Optional[str]
    direction: str
    spacy_match: List[Dict[str, Any]]

    def to_json(self):

        return {
                "name": self.name,
                "label": self.label,
                "regex": self.regex,
                "direction": self.direction,
                "spacy_match": self.spacy_match
        }


nlp = spacy.load("en_core_web_sm")


Span.set_extension("negation", default="", force=True)
Span.set_extension("temporality", default="PRESENT", force=True)


def update_entities_inside_window(start: int, end: str, entities: List[Span], label: str) -> None:
    """
    Given a window into a document and a label, updates any
    entities which fall inside the span (endpoint inclusive).
    Note that this function modifies entities in place.
    """
    if end - start == 0:
        return
    for entity in entities:
        if start <= entity.start <= end and start <= entity.end <= end:
            if label in TEMPORAL_BREAKS:
                entity._.temporality = label
            elif label in NEGATION_BREAKS:
                entity._.negation = label


def context_algorithm(substituted: List[str],
                      break_triggers: Set[str],
                      entities: List[Span]) -> None:
    max_length = 10
    window = 0
    length = len(substituted)
    i = 0
    while i < length:
        token = substituted[i]
        if isinstance(token, str):
            i += 1
            continue

        label, direction = token

        if label == "PSEUDONEG":
            i += 1
            continue
        
        if direction in {"forward", "bidirectional"} and label != "CONJ":
            # Look forward to find things in this scope
            window_max = min(length, i + max_length)
            print("going forward")
            start = i + 1
            for j in range(start, window_max):
                print(j)
                window_token = substituted[j]
                if isinstance(window_token, str):
                    window += 1
                    continue
                window_label, _ = window_token
                if window_label in break_triggers:
                    break
            # update entities
            update_entities_inside_window(start, start + window, entities, label)
            print(start, start + window, substituted[i: i+ window], label, direction)

        window = 0
        if direction in {"backward", "bidirectional"} and label != "CONJ":
            for j in range(i - 1, max(i - max_length, -1), -1):
                window_token = substituted[j]
                if isinstance(window_token, str):
                    window += 1
                    continue
                window_label, _ = window_token
                if window_label in break_triggers:
                    break
            update_entities_inside_window(i - window, i, entities, label)
            print(i - window, i, substituted[i - window: i], label, direction)

        i += 1
        window = 0

TEMPORAL = {
        'CONJ',
        'FUTURE',
        'HISTORICAL',
        'HYPOTHETICAL',
        'INDICATION', # TODO check why this is different from hypothetical
        'PSEUDONEG'
}

NEGATION = {
        'AMBIVALENT_EXISTENCE',
        'CONJ',
        'DEFINITE_EXISTENCE',
        'DEFINITE_NEGATED_EXISTENCE',
        'PROBABLE_EXISTENCE',
        'PROBABLE_NEGATED_EXISTENCE',
        'PSEUDONEG'
}

TEMPORAL_BREAKS = TEMPORAL.difference({"PSEUDONEG"})
NEGATION_BREAKS = NEGATION.difference({"PSEUDONEG"})

class NegationAndTemporalContext:
    """
    Implements "Context: An Algorithm for Determining Negation,
    Experiencer, and Temporal Status from Clinical Reports", but doesn't include
    the Experiencer one because it has lower precision and recall than Negation
    and Temporal status.

    This class sets the `._.negation` and `._.temporality` attributes of `spacy.tokens.Span`.

    Valid labels for `._.negation` are:
        '' (The default label)
        'AMBIVALENT_EXISTENCE'
        'DEFINITE_EXISTENCE'
        'DEFINITE_NEGATED_EXISTENCE'
        'PROBABLE_EXISTENCE'
        'PROBABLE_NEGATED_EXISTENCE'

    Valid labels for `._.temporality` are:
        'PRESENT' (the default)
        'FUTURE'
        'HISTORICAL'
        'HYPOTHETICAL'
        'INDICATION'
    """

    def __init__(self, vocab, lexicon_path: str):
        patterns = [ContextPattern(**json.loads(line.strip())) for line in open(lexicon_path)]

        self.pattern_to_type = {x.name: (x.label, x.direction) for x in patterns}
        self.matcher = Matcher(vocab)

        for pattern in patterns:
            self.matcher.add(pattern.name, None, pattern.spacy_match)


    def __call__(self, doc):
        matches = self.matcher(doc)

        words_temporal = [t.text for t in doc]
        words_negation = [t.text for t in doc]

        # This seems massively crude - we just overwrite any spans which have
        # already been annotated (which is why we traverse them shortest to longest).
        # However, it's the way the original algorithm works
        # (possibly their regexes only match strictly nested subspans..)
        for match in sorted(matches, key=lambda x: x[2] - x[1]):
            hash_id, start, end = match
            string_id = self.matcher.vocab.strings[hash_id]
            label, direction = self.pattern_to_type[string_id]

            print(label, direction, start, end)
            if label in NEGATION:
                for j in range(start, end):
                    words_negation[j] = (label, direction)     

            if label in TEMPORAL:
                for j in range(start, end):
                    words_temporal[j] = (label, direction)     

        print(words_negation)
        print(words_temporal)
        print("Negations: ")
        entities = list(doc.ents)
        context_algorithm(words_negation, NEGATION_BREAKS, entities)
        print("Temporal: ")
        context_algorithm(words_temporal, TEMPORAL_BREAKS, entities)

        return doc
