import pytest

from scispacy.genia_tokenizer import GeniaTokenizer
TEST_CASES = [("using a bag-of-words model", ["using", "a", "bag-of-words", "model"]),
              ("activators of cAMP- and cGMP-dependent protein", ["activators", "of", "cAMP-", "and", "cGMP-dependent", "protein"]),
              ("phorbol 12-myristate 13-acetate, caused almost", ["phorbol", "12-myristate", "13-acetate", ",", "caused", "almost"]),
              ("let C(j) denote", ["let", "C(j)", "denote"]),
              ("let (C(j)) denote", ["let", "(", "C(j)", ")", "denote"]),
              ("let C{j} denote", ["let", "C{j}", "denote"]),
              ("for the camera(s) and manipulator(s)", ["for", "the", "camera(s)", "and", "manipulator(s)"]),
              ("the (TRAP)-positive genes", ["the", "(TRAP)-positive", "genes"]),
              ("the {TRAP}-positive genes", ["the", "{TRAP}-positive", "genes"]),
              ("for [Ca2+]i protein", ["for", "[Ca2+]i", "protein"]),
              ("for pyrilamine[3H] protein", ["for", "pyrilamine[3H]", "protein"]),
              ("this is (normal) parens", ["this", "is", "(", "normal", ")", "parens"]),
              ("this is [normal] brackets", ["this", "is", "[", "normal", "]", "brackets"]),
              ("this is {normal} braces", ["this", "is", "{", "normal", "}", "braces"]),
              ("in the lan-\nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan-\n\nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan- \nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan- \n\nguage of the", ["in", "the", "language", "of", "the"]),
              ("a 28× 28 image", ["a", "28", "×", "28", "image"]),
              ("a 28×28 image", ["a", "28", "×", "28", "image"]),
              ("a 28 × 28 image", ["a", "28", "×", "28", "image"]),
              ("the neurons’ activation", ["the", "neurons", "’", "activation"]),
              ("the neurons' activation", ["the", "neurons", "'", "activation"]),
              ("H3G 1Y6", ["H3G", "1Y6"]),
              ("HFG 1Y6", ["HFG", "1Y6"]),
              ("H3g 1Y6", ["H3g", "1Y6"]),
              ("h3g 1Y6", ["h3g", "1Y6"]),
              ("h36g 1Y6", ["h36g", "1Y6"]),
              ("h3gh 1Y6", ["h3gh", "1Y6"]),
              ("h3g3 1Y6", ["h3g3", "1Y6"]),
              ("interleukin (IL)-mediated", ["interleukin", "(IL)-mediated"]),
              ("This can be seen in Figure 1D. Therefore", ["This", "can", "be", "seen", "in", "Figure", "1D", ".", "Therefore"]),
              ("This can be seen in Figure 1d. Therefore", ["This", "can", "be", "seen", "in", "Figure", "1d", ".", "Therefore"]),
              ("This is a sentence.", ["This", "is", "a", "sentence", "."]),
              ("result of 1.345 is good", ["result", "of", "1.345", "is", "good"]),
              ("This sentence ends with a single 1.", ["This", "sentence", "ends", "with", "a", "single", "1", "."]),
              ("This sentence ends with a single 1. This is the next sentence.", ["This", "sentence", "ends", "with", "a", "single", "1", ".", "This", "is", "the", "next", "sentence", "."]),
              ("sec. secs. Sec. Secs. fig. figs. Fig. Figs. eq. eqs. Eq. Eqs. no. nos. No. Nos. al. .", ["sec.", "secs.", "Sec.", "Secs.", "fig.", "figs.", "Fig.", "Figs.", "eq.", "eqs.", "Eq.", "Eqs.", "no.", "nos.", "No.", "Nos.", "al.", "."]),
              ("in the Gq/G11 protein", ["in", "the", "Gq/G11", "protein"]),
              ("in the G1/G11 protein", ["in", "the", "G1/G11", "protein"]),
              ("in the G1/11 protein", ["in", "the", "G1/11", "protein"]),
              ("in the Gq/11 protein", ["in", "the", "Gq/11", "protein"]),
             ]

@pytest.mark.parametrize('text,expected_tokens', TEST_CASES)
def test_custom_tokenization(combined_all_model_fixture, remove_new_lines_fixture, text, expected_tokens):
    text = remove_new_lines_fixture(text)
    doc = combined_all_model_fixture(text)
    tokens = [t.text for t in doc]
    assert tokens == expected_tokens

@pytest.mark.parametrize('text,expected_tokens', TEST_CASES)
def test_genia_custom_tokenization(combined_all_model_fixture, remove_new_lines_fixture, text, expected_tokens):
    text = remove_new_lines_fixture(text)
    tokenizer = GeniaTokenizer(combined_all_model_fixture.vocab)
    doc = tokenizer(text)
    tokens = [t.text for t in doc]
    assert tokens == expected_tokens
