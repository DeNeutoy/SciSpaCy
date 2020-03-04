from typing import List
from collections import defaultdict
from spacy import displacy


def visualise_doc(doc):
    displacy.render(doc, style="dep", options={"distance": 120})
    displacy.render(doc, style="ent", options={"distance": 120})


def visualise_subtrees(doc, subtrees):

    words = [{"text": t.text, "tag": t.pos_} for t in doc]

    if not isinstance(subtrees[0], list):
        subtrees = [subtrees]

    for subtree in subtrees:
        arcs = []

        tree_indices = set(subtree)
        for index in subtree:

            token = doc[index]
            head = token.head
            if token.head.i == token.i or token.head.i not in tree_indices:
                continue

            else:
                if token.i < head.i:
                    arcs.append(
                        {
                            "start": token.i,
                            "end": head.i,
                            "label": token.dep_,
                            "dir": "left",
                        }
                    )
                else:
                    arcs.append(
                        {
                            "start": head.i,
                            "end": token.i,
                            "label": token.dep_,
                            "dir": "right",
                        }
                    )
        print("Subtree: ", subtree)
        displacy.render(
            {"words": words, "arcs": arcs},
            style="dep",
            options={"distance": 120},
            manual=True,
        )


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


def check_for_non_trees(rules: List[List[str]]):

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

    return root, parent_to_children


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

    ret = check_for_non_trees(rules)

    if ret is None:
        return None
    else:
        root, parent_to_children = ret

    def add_node(parent: str, pattern: List):

        for (rel, child) in parent_to_children[parent]:

            # First, we add the specification that we are looking for
            # an edge which connects the child to the parent.
            node = {
                "SPEC": {"NODE_NAME": child, "NBOR_RELOP": ">", "NBOR_NAME": parent}
            }

            # DANGER we can only have these options IF we also match ORTH below, otherwise it's torturously slow.
            # Now, we specify what attributes we want this _token_
            # to have - in this case, we want to match a certain dependency
            # relation specifically.
            if rel == "dep":
                # The generic "dep" relation usually indicates a parser error.
                # In this case we just settle for there being an edge.
                token_pattern = {}

            # elif rel in {"amod", "compound"}:
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
