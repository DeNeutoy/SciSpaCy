import os
import spacy
import sys
import shutil
import copy
import argparse
import json
import random
import itertools
import gzip

from pathlib import Path
from tqdm import tqdm
from spacy import util

from timeit import default_timer as timer
from spacy.cli.train import _get_progress
from spacy.vocab import Vocab
from spacy.gold import GoldCorpus
from spacy.language import Language
from wasabi import Printer
import srsly

from thinc.neural.util import prefer_gpu
from spacy._ml import Tok2Vec, create_default_optimizer
from spacy.cli.pretrain import create_pretraining_model, ProgressTracker, make_docs, make_update

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from scispacy.custom_sentence_segmenter import combined_rule_sentence_segmenter



def abstract_iterator(directory_path: str):
    print(directory_path)
    print(os.listdir(directory_path))
    print(os.getcwd())

    files = os.listdir(directory_path)
    random.shuffle(files)
    for file_path in files:
        for line in gzip.open(os.path.join(directory_path, file_path)):
            paper_dict = json.loads(line.strip())
            yield {"text": paper_dict["paperAbstract"]}


def pretrain(data_dir: str,
             model_output_dir: str,
             model_path: str = None,
             iterations: int = 10):

    msg = Printer()
    has_gpu = prefer_gpu()
    msg.info("Using GPU" if has_gpu else "Not using GPU")

    if model_path is not None:
        Language.factories['combined_rule_sentence_segmenter'] = lambda nlp, **cfg: combined_rule_sentence_segmenter
        nlp = spacy.load(model_path)
    else:
        lang_class = util.get_lang_class('en')
        nlp = lang_class()

    if model_path is not None:
        meta = nlp.meta
    else:
        meta = {}
        meta["lang"] = "en"
        meta["pipeline"] = ["tagger", "parser"]
        meta["name"] = "scispacy_core_web_sm"
        meta["license"] = "CC BY-SA 3.0"
        meta["author"] = "Allen Institute for Artificial Intelligence"
        meta["url"] = "allenai.org"
        meta["sources"] = ["OntoNotes 5", "Common Crawl", "GENIA 1.0"]
        meta["version"] = "1.0.0"
        meta["spacy_version"] = ">=2.0.18"
        meta["parent_package"] = "spacy"
        meta["email"] = "ai2-info@allenai.org"

    util.fix_random_seed(util.env_opt("seed", 24543))


    output_dir = Path(model_output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
        msg.good("Created output directory")

    texts = list(abstract_iterator(data_dir))

    use_vectors = True
    dropout = 0.2
    pretrained_vectors = None if not use_vectors else nlp.vocab.vectors.name
    model = create_pretraining_model(
        nlp,
        Tok2Vec(
            width=96,
            embed_size=2000,
            conv_depth=4,
            pretrained_vectors=pretrained_vectors,
            bilstm_depth=0,  # Requires PyTorch. Experimental.
            cnn_maxout_pieces=3,  # You can try setting this higher
            subword_features=True,  # Set to False for Chinese etc
        ),
    )
    optimizer = create_default_optimizer(model.ops)
    tracker = ProgressTracker(frequency=10000)
    msg.divider("Pre-training tok2vec layer")
    row_settings = {"widths": (3, 10, 10, 6, 4), "aligns": ("r", "r", "r", "r", "r")}
    msg.row(("#", "# Words", "Total Loss", "Loss", "w/s"), **row_settings)

    save_checkpoints = set()
    for epoch in range(iterations):
        for batch in util.minibatch_by_words(
            ((text, None) for text in texts), size=1000
        ):
            docs = make_docs(nlp, [text for (text, _) in batch])
            loss = make_update(model, docs, optimizer, drop=dropout)
            progress = tracker.update(epoch, loss, docs)
            if progress:
                msg.row(progress, **row_settings)
                step_check = int(tracker.words_per_epoch[epoch] / (20 ** 8))
                if step_check not in save_checkpoints and step_check > 0:
                    save_checkpoints.add(step_check)
                    with model.use_params(optimizer.averages):
                        with (output_dir / f"model_{str(tracker.words_per_epoch[epoch])}.bin").open("wb") as file_:
                            file_.write(model.tok2vec.to_bytes())

                        log = {
                            "nr_word": tracker.nr_word,
                            "loss": tracker.loss,
                            "epoch_loss": tracker.epoch_loss,
                            "epoch": epoch,
                        }
                        with (output_dir / "log.jsonl").open("a") as file_:
                            file_.write(srsly.json_dumps(log) + "\n")


        with model.use_params(optimizer.averages):
            with (output_dir / f"model{str(epoch)}.bin").open("wb") as file_:
                file_.write(model.tok2vec.to_bytes())

            log = {
                "nr_word": tracker.nr_word,
                "loss": tracker.loss,
                "epoch_loss": tracker.epoch_loss,
                "epoch": epoch,
            }
            with (output_dir / "log.jsonl").open("a") as file_:
                file_.write(srsly.json_dumps(log) + "\n")
        tracker.epoch_loss = 0.0
        # Reshuffle the texts if texts were loaded from a file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        help="Path to the json formatted training data"
    )

    parser.add_argument(
        '--model_path',
        default=None,
        help="Path to the spacy model to load"
    )

    parser.add_argument(
        '--model_output_dir',
        help="Path to the directory to output the trained models to"
    )
    parser.add_argument(
        '--iterations',
        default=10,
        help="Number of iterations to pretrain for."
    )


    args = parser.parse_args()
    pretrain(args.data_dir,
             args.model_output_dir,
             args.model_path,
             args.iterations)
