# coding: utf8
from __future__ import unicode_literals

import os
from typing import Dict, Any
import hug
from hug_middleware_cors import CORSMiddleware
import spacy

MODELS = {
    'en_core_web_sm': spacy.load('en_core_web_sm'),
}

def get_model_desc(nlp, model_name):
    """Get human-readable model name, language name and version."""
    lang_cls = spacy.util.get_lang_class(nlp.lang)
    lang_name = lang_cls.__name__
    model_version = nlp.meta['version']
    return '{} - {} (v{})'.format(lang_name, model_name, model_version)


def collapse_noun_phrases(document: spacy.tokens.Doc) -> None:
    for np in list(document.noun_chunks):
        np.merge(tag=np.root.tag_, lemma=np.root.lemma_,
                    ent_type=np.root.ent_type_)

@hug.get('/models')
def models():
    return {name: get_model_desc(nlp, name) for name, nlp in MODELS.items()}



@hug.post('/dep')
def dep(text: str, model: str, collapse_punctuation: bool=False,
        collapse_phrases: bool=False):
    """Get dependencies for displaCy visualizer."""
    nlp = MODELS[model]
    doc = nlp(text)
    if collapse_phrases:
        for np in list(doc.noun_chunks):
            np.merge(tag=np.root.tag_, lemma=np.root.lemma_,
                     ent_type=np.root.ent_type_)
    options = {'collapse_punct': collapse_punctuation}
    print(doc)
    return spacy.displacy.parse_deps(doc, options)


@hug.post('/ent')
def ent(text: str, model: str):
    """Get entities for displaCy ENT visualizer."""
    nlp = MODELS[model]
    doc = nlp(text)
    return [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_}
            for ent in doc.ents]


dir_path = os.path.dirname(os.path.realpath(__file__))
build_dir = dir_path + '/../demo/build'

@hug.static('/')
def static_root():
    return (build_dir,)

@hug.static('/spacy-parser')
def static_model():
    return (build_dir,)

@hug.static('/static/js')
def static_js():
    return (build_dir + "/static/js",)


if __name__ == "__main__":
    import waitress
    app = hug.API(__name__)
    app.http.add_middleware(CORSMiddleware(app))
    waitress.serve(__hug_wsgi__, port=8080)
