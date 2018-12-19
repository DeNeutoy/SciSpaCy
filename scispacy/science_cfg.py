
from typing import Dict, List, Tuple
import random

class ContextFreeGrammar:
    """
    A simple context free grammar.

    Parameters
    ----------
    production_rules: Dict[str, List[Tuple[str]]]
        A dictionary mapping non-terminals to production rules
        represented as tuples of either non-terminal or terminal symbols.
    """
    def __init__(self, production_rules: Dict[str, List[Tuple[str]]]):
        self.prod = production_rules

    def gen_random(self, root: str):
        """ Generate a random sentence from the
            grammar, starting with the given
            symbol.
        """
        sentence = []

        # select one production of this symbol randomly
        rand_prod = random.choice(self.prod[root])
        for symbol in rand_prod:
            # for non-terminals, recurse
            if symbol in self.prod:
                sentence = sentence + self.gen_random(symbol)
            else:
                sentence.append(symbol)

        return sentence


science_grammar = {}

science_grammar["root"] = [("(", "descriptor", "paper_or_content", ")")]

science_grammar["descriptor"] = [("see ",), ("for ", "example", ", "), ("demonstrated ", "by ")]

science_grammar["paper_or_content"] = [("object_with_punct", "identifier"), ("paper", )]
science_grammar["object_with_punct"] = [("object", "maybe_punct")]
science_grammar["object"] = [("Table",), ("Figure",), ("Equation",), ("Section",), ("abreviation",)]
science_grammar["identifier"] = [("number", "letter", "maybe_punct"),
                                 ("number", "-", "letter", "maybe_punct"),
                                 ("number", "maybe_punct"),
                                 ("letter", "maybe_punct")]


science_grammar["paper"] = [("(", "name", "maybe_et_al", ", ", "year", ")"),
                            ("name", "maybe_et_al", ", ", "year"),
                            ("[", "number", "]"),
                            ("[", "number", "letter", "]"),
                            ("(", "number", ")"),
                            ("(", "number", "letter", ")")]
science_grammar["name"] = [("Peters", ), ("Neumann", ), ("Agerwal",)]
science_grammar["year"] = [(f"{str(x)}",) for x in range(1920, 2018)]
science_grammar["maybe_et_al"] = [(" et al.",), ("")]

science_grammar["abreviation"] = [("Eq",), ("eq",), ("Fig",), ("fig",), ("Sec",)]
science_grammar["number"] = [(f"{str(x)}",) for x in range(1, 10)]
science_grammar["letter"] = [(f"{str(x)}",) for x in ("A", "B", "C", "D", "E", "F", "G",
                                                      "a", "b", "c", "d", "e", "f", "g")]
science_grammar["maybe_punct"] = [(". ",), ("."), (" ",)]

grammar = ContextFreeGrammar(science_grammar)


for i in range(50):

    print("".join(grammar.gen_random("root")))