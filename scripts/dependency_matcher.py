from typing import Dict, List
from collections import defaultdict
import sys
import os
import argparse
import json

import spacy
from spacy.matcher import DependencyMatcher
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from scispacy.util import WhitespaceTokenizer

PTB_BRACKETS = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LCB-": "{",
        "-RCB-": "}",
        "-LSB-": "[",
        "-RSB-": "]",
}

def clean_and_parse(sent: str, nlp):

    tokens = sent.strip().split(" ")

    new = []

    for token in tokens:
        new_token = PTB_BRACKETS.get(token, None)
        if new_token is None:
            new.append(token)
        else:
            new.append(new_token)

    return nlp(" ".join(new))


def load_distributions(distribution_path: str) -> Dict[str, Dict[str, float]]:

    with open(distribution_path) as f:
        headers = next(f).strip().split("\t")[1:]

        distributions = {}
        for line in f:
            line = line.strip().split("\t")
            distributions[line[0]] = {name: float(value) for name, value in zip(headers, line[1:])}

    return distributions

def parse_dep_path(dep_string: str):

    rules = [rule.split("|") for rule in dep_string.split(" ")]

    for triple in rules:

        if triple[0] in PTB_BRACKETS:
            triple[0] = PTB_BRACKETS[triple[0]]

        if triple[2] in PTB_BRACKETS:
            triple[2] = PTB_BRACKETS[triple[2]]

        if triple[1] == "nsubj:xsubj":
            triple[1] = "nsubj"

        if triple[1] == "nsubjpass:xsubj":
            triple[1] = "nsubjpass"
    return rules

def construct_pattern(rules: List[List[str]]):
    """
    Idea: add patterns to a matcher designed to find a subtree in a spacy dependency tree.
    Rules are strictly of the form "CHILD --rel--> PARENT". To build this up, we add rules
    in DFS order, so that the parent nodes have already been added to the dict for each child
    we encounter.
    """
    # Step 1: Build up a dictionary mapping parents to their children
    # in the dependency subtree. Whilst we do this, we check that there is
    # a single node which has only outgoing edges.

    if "dep" in {rule[1] for rule in rules}:
        return None

    parent_to_children = defaultdict(list)
    seen = set()
    has_incoming_edges = set()
    for (parent, rel, child) in rules:
        seen.add(parent)
        seen.add(child)
        has_incoming_edges.add(child)
        if parent == child:
            return None
        parent_to_children[parent].append((rel, child))

    # Only accept strictly connected trees.
    roots = seen.difference(has_incoming_edges)
    if len(roots) != 1:
        return None

    root = roots.pop()
    seen = {root}

    # Step 2: check that the tree doesn't have a loop:
    def contains_loop(node):
        has_loop = False
        for (_, child) in parent_to_children[node]:
            if child in seen:
                return True
            else:
                seen.add(child)
                has_loop = contains_loop(child)
            if has_loop:
                break

        return has_loop

    if contains_loop(root):
        return None

    def add_node(parent: str, pattern: List):
        
        for (rel, child) in parent_to_children[parent]:

            # First, we add the specification that we are looking for
            # an edge which connects the child to the parent.
            node = {
                "SPEC": {
                    "NODE_NAME": child,
                    "NBOR_RELOP": ">",
                    "NBOR_NAME": parent},
            }

            # DANGER we can only have these options IF we also match ORTH below, otherwise it's torturously slow.
            # Now, we specify what attributes we want this _token_
            # to have - in this case, we want to match a certain dependency
            # relation specifically.
            if rel == "dep":
                # The generic "dep" relation usually indicates a parser error.
                # In this case we just settle for there being an edge.
                token_pattern = {}

            #elif rel in {"amod", "compound"}:
            #    # TODO double check that this is symmetric
            #    # These are inconsistently annotated, so if either occur we accept either.
            #    token_pattern = {"DEP": {"IN": ["amod", "compound"]}}
            else:
                # this is the default case, where we need to match the relation exactly.
                token_pattern = {"DEP": rel}

            if child not in {"START_ENTITY", "END_ENTITY"}:
                token_pattern["ORTH"] = child

            else:
                token_pattern["ENT_TYPE"] = {"NOT_IN": [""]}
                token_pattern["POS"] = "NOUN"

            node["PATTERN"] = token_pattern

            pattern.append(node)
            add_node(child, pattern)

    pattern = [{"SPEC": {"NODE_NAME": root}, "PATTERN": {"ORTH": root}}]
    add_node(root, pattern)

    assert len(pattern) < 20
    return pattern


def read_examples(examples_path: str):
    """
    Read data into json blobs.
    """
    data_headers = ["pmid", "sent", "ent1", "ent1_offset", "ent2", "ent2_offset",
                    "ent1_raw", "ent2_raw", "ent1_canonical", "ent2_canonical", 
                    "ent1_type", "ent2_type", "dep", "sent"]

    with open(examples_path) as data:
        lines = [{k:v for k, v in zip(data_headers, line.strip().split("\t") )}
                 for line in data]

    return lines

def main(examples_path: str, out_path: str):
    data = read_examples(examples_path)
    nlp = spacy.load("en_core_sci_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


    good_patterns = {}
    good_pattern_subtrees = []
    bad_patterns = {}
    bad_pattern_subtrees = []
    for i, example in enumerate(data):

        if i % 10 == 0:
            print(f"Done {i} examples.")
        dep_pattern = example["dep"]
        subtree = parse_dep_path(dep_pattern)

        pattern = construct_pattern(subtree)
        if pattern is None:
            continue

        matcher = DependencyMatcher(nlp.vocab)
        matcher.add("pattern1", None, pattern)

        #doc = clean_and_parse(example["sent"], nlp)
        #matches = matcher(doc)
        #match = matches[0]

        #if match[1]:
        if True:
            if dep_pattern.lower() not in good_patterns:
                good_patterns[dep_pattern.lower()] = 1
                good_pattern_subtrees.append({"name": dep_pattern.lower(), "pattern": pattern})
            else:
                good_patterns[dep_pattern.lower()] += 1

        else:
            if dep_pattern.lower() not in bad_patterns:
                bad_patterns[dep_pattern.lower()] = 1
                bad_pattern_subtrees.append({"name": dep_pattern.lower(), "pattern": pattern})
            else:
                bad_patterns[dep_pattern.lower()] += 1



    with open(out_path, "w+") as out:
        for patt in good_pattern_subtrees:
            patt["freq"] = good_patterns[patt["name"]]

            out.write(json.dumps(patt) + "\n")

    with open("bad_" + out_path, "w+") as out:
        for patt in bad_pattern_subtrees:
            patt["freq"] = bad_patterns[patt["name"]]

            out.write(json.dumps(patt) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--examples',
            help="Path to the file containing the sentences."
    )
    parser.add_argument(
            '--out',
            help="Path to the output file."
    )

    args = parser.parse_args()
    main(args.examples, args.out)
