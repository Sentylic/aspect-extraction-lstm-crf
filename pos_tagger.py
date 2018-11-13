#!/usr/bin/python
from stanford_corenlp_pywrapper import CoreNLP

proc = CoreNLP("pos", corenlp_jars=["./stanford-corenlp-full-2018-10-05/*"])


def pos_tagger(string):
    return tokenize(string)[0]


def tokenize(string):
    parse_ret = proc.parse_doc(string)
    tokens = []
    pos = []

    sents = parse_ret["sentences"]
    for sent in sents:
        tokens.extend(sent["tokens"])
        pos.extend(sent["pos"])
        assert len(sent["tokens"])==len(sent["pos"])
    return pos, tokens


if __name__ == "__main__":
    temp = tokenize("I'll be back for sure.".strip())
    print temp
