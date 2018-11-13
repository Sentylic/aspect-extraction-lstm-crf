#!/usr/bin/python
from stanford_corenlp_pywrapper import CoreNLP

proc = CoreNLP("pos", corenlp_jars=["./stanford-corenlp-full-2018-10-05/*"])


def tokenize(string):
    parse_ret = proc.parse_doc(string)
    ret_l = []
    sents = parse_ret["sentences"]
    for sent in sents:
        ret_l.extend(sent["tokens"])
    return ret_l

if __name__ == "__main__":
    print tokenize("I`ll be back for sure.")