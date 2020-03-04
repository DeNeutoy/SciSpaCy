ty Rules #
# ------------- #


def expand_to_sub_graph() -> ExpansionRule:
    first_level = expand_to(child(edge.any(), ~node.in_expansion_span()))
    second_level = expand_to(child(edge.any()))
    return try_expansion(first_level) >> try_expansion(second_level)


# ------------- #
# SUMMARY RULES #
# ------------- #


def _9_mark() -> ExpansionRule:
    """
    chunk: VP
    priority: low
    example test positive: I am glad we were able [mark to] [capture answer] this without getting snotty .
    example test negative: you agree [mark that] the genealogics site should be [capture credited]
    """
    return expand_to(child(edge.eq("mark"), tag.one_of("TO", "PART")))


def _10_advmod() -> ExpansionRule:
    """
    chunk: VP/ADJP/RB
    priority: middle
    example test positive: this article will almost [advmod certainly] be [head deleted]
    Example test negative: I " m glad you are not [capture lying about it [advmod now] , [advmod though ].
    IN and NN are not included because with advmod they are mostly predicates and the advmod modifies the entire
    clause (advs like however, therefore...).
    """
    # question: do we want wh-adverbs?
    return only_when(
        ~tag.one_of("NN", "NNS", "IN")
        & ~(index.of(child(edge.startswith("nsubj"))).greater_than(index.of(child(edge.eq("advmod")))))
    ) >> expand_to(child(edge.eq("advmod"), index.of(here()).lesser_than(index.of(previous()))))


def _14_parent_mwe() -> ExpansionRule:
    """
    chunk: NP
    priority: high
    example test: Col. Jefferson J. [capture DeBlanc] [parent_compound Sr.] , 86 , a retired Marine Corps fighter
    """
    return expand_to(parent(edge.one_of("compound", "mwe", "flat", "fixed")))  # flat and fixed are scispacy-related


def _15_entity_appos() -> ExpansionRule:
    """
    chunk: NP
    priority: low - TBD
    example test: Sodium-glucose cotransporter 2 [capture inhibitor] ( [appos SGLT2i] )
                  is a new class of antidiabetic therapy
    """
    return expand_to(child(edge.eq("appos"), entity.exists()))


def _16_scispacy_nmod_of_equiv() -> ExpansionRule:
    """
    chunk: NP (a scispacy version of nmod:of)
    priority: low
    example test: fluoride  will  reduce  the  {capture IQ}  [$ of]  our  [^ children]
    """
    return expand_to(grand_child(edge.eq("prep"), edge.eq("pobj"), lemma.eq("of")))


def _1_general() -> ExpansionRule:
    """
    expands to obligatory elements even if they are part of conjunction.
    """
    return expand_to(
        child(edge.one_of("compound", "mwe", "det", "nummod", "aux", "auxpass", "poss", "compound:prt", "prt", "neg"))
    )


def _2_general_left_edge() -> ExpansionRule:
    """
    covers rules 3 and 13 above
    captures a child only if it appears to the left of the current edge.
    """
    return expand_to(
        child(edge.one_of("amod", "nmod:npmod", "npadvmod"), index.of(here()).lesser_than(index.of(previous())))
    )


def _3_general_tbd() -> ExpansionRule:
    """
    covers 5,8,11,12 above
    labels I'm not sure about. not included in ruleset.
    neg - for VPs it should be captured, but when connected to nouns it is mostly negating the verb whose
    object is the noun (e.g he sees [no reason] to be concerned).
    acl/xcomp/cop - tbd if the complement is a natural part of the head chunk.
    """
    return expand_to(child(edge.one_of("neg", "acl:to", "acl_to", "xcomp", "cop")))


def _4_tense() -> ExpansionRule:
    return expand_to(
        parent(edge.eq("xcomp"), (word.one_of("going", "gon", "used") & ~has_outgoing_edge.one_of("dobj", "auxpass")))
    )


##################
# subordinated clauses
##################


def _1_sub_wdt() -> ExpansionRule:
    """
    chunk: NP
    priority: low
    example test: I can do [wdt whatever] I want]
    """
    return only_when(lemma.startswith("w") & lemma.endswith("ever")) >> expand_to(
        parent(edge.any(), index.of(here()).greater_than(index.of(previous())))
    )


def _1_relcl_intolerant_to_ccomp() -> ExpansionRule:
    """
    chunk: NP
    expanding to relcl children only if they are nsubj or dobj
    """
    first_level = expand_to(child(edge.eq("acl:relcl"), ~node.has(child(edge.one_of("ccomp", "acl:relcl")))))
    second_level = expand_to(
        child(edge.any(), (~node.has(child(edge.one_of("ccomp", "acl:relcl"))) & ~node.in_expansion_span()))
    )

    third_level = expand_to(child(edge.any(), (~node.has(child(edge.one_of("ccomp", "acl:relcl"))))))

    return (
        only_when(~has_outgoing_edge.eq("nmod:poss"))
        >> first_level
        >> try_expansion(second_level)
        >> try_expansion(third_level)
    )


def _2_relcl_tolerant_to_one_ccomp() -> ExpansionRule:
    """
    chunk: NP
    expanding to relcl children only if they are nsubj or dobj
    """

    first_level = expand_to(child(edge.eq("acl:relcl")))
    second_level = expand_to(
        child(edge.any(), (~node.has(child(edge.one_of("ccomp", "acl:relcl"))) & ~node.in_expansion_span()))
    )
    third_level = expand_to(child(edge.any(), (~node.has(child(edge.one_of("ccomp", "acl:relcl"))))))

    return (
        only_when(~has_outgoing_edge.eq("nmod:poss"))
        >> first_level
        >> try_expansion(second_level)
        >> try_expansion(third_level)
    )


def _3_parent_relcl_permissive() -> ExpansionRule:
    """
    chunk: NP
    example: there are shelves of [capture books] that [xcomp mention] his association with Balanchine.
    currently expands to the immediate children of the relative clause's head and their children (but not further).
    """

    first_level = expand_to(child(edge.eq("acl:relcl")))
    second_level = expand_to(child(edge.any(), ~node.in_expansion_span()))
    third_level = expand_to(child(edge.any()))

    return (
        only_when(~has_outgoing_edge.eq("nmod:poss"))
        >> first_level
        >> try_expansion(second_level)
        >> try_expansion(third_level)
    )


def _3_parent_relcl_2_levels_with_obligatory_args() -> ExpansionRule:
    """
    chunk: NP
    expanding to relcl children only if they are nsubj or dobj
    """
    first_level = expand_to(child(edge.eq("acl:relcl")))
    expand_nsubj_dobj = expand_to(child(edge.one_of("nsubj", "dobj")))

    return only_when(~has_outgoing_edge.eq("nmod:poss")) >> first_level >> expand_nsubj_dobj >> expand_nsubj_dobj


def _4_parent_relcl_2_levels_without_long_complements() -> ExpansionRule:
    """
    chunk: NP
    expanding to relcl children only if they are nsubj or dobj
    """

    first_level = expand_to(child(edge.eq("acl:relcl")))
    expand_sub_clause = expand_to(child((~edge.one_of("ccomp", "xcomp", "acl:relcl") | ~edge.startswith("advcl"))))

    return only_when(~has_outgoing_edge.eq("nmod:poss")) >> first_level >> expand_sub_clause >> expand_sub_clause


def _1_sub_xcomp_general() -> ExpansionRule:
    """
    chunk: VP
    example: Islamic militants could [head prove] [xcomp useful] in pressuring its historic rival India
    """
    return expand_to(child(edge.eq("xcomp"), (tag.startswith("JJ") | tag.startswith("NN"))))


def _2_sub_xcomp_jj() -> ExpansionRule:
    """
    chunk: VP
    example: Blackburn Rovers are [capture delighted] to [xcomp announce] the appointment of Sam Allardyce as manager
    """
    # return only_when(tag.startswith("JJ")) >> expand_to_child(edge.eq("xcomp"))
    return only_when(has_incoming_edge.eq("xcomp")) >> expand_to(parent(edge.eq("xcomp"), tag.startswith("JJ")))


def _6_sub_xpos_permissive() -> ExpansionRule:
    """
    chunk: VP
    priority: low - way too permissive and broad.
    positive example: her grandson is [head seeking] to [xcomp fill] out the rest of her sixth term
    negative example: The Yaroslav Mudry frigate is [head heading] to a Baltic Sea port to [xcomp begin] sea-trials
    """
    return expand_to(child(edge.eq("xpos")))


def _1_acl_deep() -> ExpansionRule:
    return expand_to(child(edge.one_of("acl_of", "acl_to"))) >> expand_to_sub_graph()


def _2_acl_shallow() -> ExpansionRule:
    return expand_to(child(edge.one_of("acl_of", "acl_to")))


##################
# prepositions - there are 187 types of nmod, whereas 10 of them make 80% of the nmod instances in tacred trainset.
# these are: 'nmod_of',  'nmod:poss','nmod_in','nmod_to','nmod_for','nmod_on','nmod_with','nmod_at',
#            'nmod_from','nmod_by'
# Of them nmod_in, nmod_for, nmod_on, nmod_with (high ambiguity)  were found irrelevant for expansion.
##################


def _1_nmod_high_priority() -> ExpansionRule:
    """
    mandatory nmod types
    chunk: NP
    """
    return only_when(~tag.startswith("VB")) >> expand_to(
        child(
            edge.one_of(
                "nmod:poss", "poss", "nmod_of", "nmod:of", "nmod_per", "nmod:per"  # nmod:of/:per and poss are scispacy
            )
        )
    )


def _2_nmod_medium_priority() -> ExpansionRule:
    """
    nmod_to:
    adj mostly "due to..."
    nouns mostly contrast to/response to and roles (UK ambassador to the states)
    """
    return only_when((~tag.one_of("CD") | ~tag.startswith("VB"))) >> expand_to(child(edge.one_of("nmod_to")))


def _3_nmod_low_priority() -> ExpansionRule:
    """
    questionable nmod types
    included:
    nmod_at: usually nice-to-have-but-not-mandatory extra info'
            (she focused her efforts on her [capture job] at [nmod_at Elle])
            plus some phrases/idioms (One [capture Step] at a [nmod_at Time])
    nmod_on: marginally relevant. most instances are date (ruled out by ~entity.one_of("DATE")).
    excluded:
    nmod_in: usually attached to copular predicates rather than serve as noun modifiers.
            (Manning was prime [capture minister] in [nmod_in 1991])
    nmod_for: usually attached to copular predicates rather than serve as noun/adj modifiers
            (their support was [capture critical] for their [nmod_for business])
    """
    return only_when(~tag.startswith("VB")) >> expand_to(
        child(edge.one_of("nmod_at", "nmod_on", "nmod_about", "nmod_from", "nmod_by"), ~entity.one_of("DATE"))
    )


def expand_to_entire_subgraph() -> ExpansionRule:
    return expand_to_recursively(child(edge.any()))


def expand_to_selective_subgraph() -> ExpansionRule:
    """
    the elimination of nsubj, mark, cop, case etc. is to have the expanded capture as it own without functional items
    that source in a parent.
    """
    return expand_to_recursively(
        child(~edge.one_of("dep", "acl:relcl", "nsubj", "case", "mark", "cop", "advmod", "ref"), ~tag.eq("WRB"))
    )


def expand_on_conjuncts() -> ExpansionRule:
    return expand_to(child(edge.startswith("conj")))


"""
Groups of rules by strategies
"""

obligatory_expansions = [
    _1_general(),  # "compound" (Supreme Court), "mwe" (more than), "det" (the cat), "nummod" (67 years)
    # "aux" (is running), "auxpass" (is troubled), "compound:prt" (shut down),
    _1_nmod_high_priority(),  # nmod:poss (their rights), nmod_of (levels of magnesium), nmod_per (two games per team)
    _2_general_left_edge(),  # amod (cute cat) nmod:npmod (a little early), npadvmod (relevant to suppai)
    _14_parent_mwe(),  # roughly same as #1 but when capturing the child
    _4_tense(),
]

optional_general_purpose_expansions: List[ExpansionRule] = [
    _9_mark(),  # to (I want to know)
    _10_advmod(),  # e.g. really hungry - sometimes high priority but also very noisy
    _15_entity_appos(),  # e.g. the american singer Madonna
    _1_sub_xcomp_general(),  # could prove useful
    _2_sub_xcomp_jj(),  # (e.g. "unable to read" when `read` is captured) debatable priority -
]

optional_nominal_complements = [
    _2_nmod_medium_priority(),  # currently only nmod_to (the Fund 's response to the global crisis)
    _3_nmod_low_priority(),  # "nmod_at" (one step at a time), "nmod_on" (a statement on the website),
    _1_relcl_intolerant_to_ccomp(),  # books that mention his association with Balanchine.
    _1_sub_wdt(),  # whatever you want
    # "nmod_about" (the rumors about the Kardashians), "nmod_from" (the girl from Ipanema),
    # "nmod_by" (A response by the Association)
    _2_acl_shallow(),
]

permissive_rules = [
    expand_to_selective_subgraph(),
    _2_relcl_tolerant_to_one_ccomp(),
    _3_parent_relcl_permissive(),
    _3_parent_relcl_2_levels_with_obligatory_args(),
    _1_acl_deep(),
    # _6_sub_xpos_permissive()
]


def recursive_expand(rules: List[ExpansionRule]) -> ExpansionRule:
    return repeat_until_expansion_is_stable(for_each_span_location(try_all(rules)))


STRATEGY_NAME_TO_EXPANSION_RULE: Dict[str, ExpansionRule] = {
    "__default__": expand_to_full_entity(),
    "E": expand_to_full_entity(),
    "U": recursive_expand(obligatory_expansions),
    "UA": recursive_expand(obligatory_expansions + optional_general_purpose_expansions),
    "UB": recursive_expand(obligatory_expansions + optional_nominal_complements),
    "ALL": recursive_expand(obligatory_expansions + optional_general_purpose_expansions + optional_nominal_complements),
    "SUBG": recursive_expand([expand_to_entire_subgraph()]),
    "U+C": recursive_expand(obligatory_expansions + [expand_on_conjuncts()]),
    "UA+C": recursive_expand(obligatory_expansions + optional_general_purpose_expansions + [expand_on_conjuncts()]),
    "UB+C": recursive_expand(obligatory_expansions + optional_nominal_complements + [expand_on_conjuncts()]),
    "ALL+C": recursive_expand(
        obligatory_expansions
        + optional_general_purpose_expansions
        + optional_nominal_complements
        + [expand_on_conjuncts()]
    ),
    "SUBG+C": recursive_expand([expand_to_entire_subgraph()] + [expand_on_conjuncts()]),
    "RSUBG": recursive_expand(
        permissive_rules + obligatory_expansions + optional_general_purpose_expansions + optional_nominal_complements
    ),
}
