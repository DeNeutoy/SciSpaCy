import json

from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc

from scispacy.file_cache import cached_path


class RelationExtractor:
    def __init__(self, vocab, patterns_file: str, min_freq: int):

        self.matcher = DependencyMatcher(vocab)

        patterns = []
        for line in open(cached_path(patterns_file)):
            pattern = json.loads(line.strip())
            patterns.append(pattern)

        patterns = sorted(patterns, key=lambda x: -x["freq"])

        i = 0
        for pattern in patterns:
            if pattern["freq"] < min_freq:
                break
            self.matcher.add(pattern["name"], None, pattern["pattern"])
            i += 1

        print(f"Loaded {i} patterns")
        self.matches = None

    def __call__(self, doc: Doc) -> Doc:

        matches = self.matcher(doc)

        self.matches = matches
        return doc
