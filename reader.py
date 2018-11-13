import xml.etree.ElementTree as ET
# from tokenizer import tokenize
import pickle
import numpy as np
from pos_tagger import tokenize


def tokenize_data(file_path, output):
    data = ET.parse(file_path)
    reviews = data.getroot()
    # root.tag
    # root.attrib

    dataset = []
    pos_set = []

    for review in reviews:
        sentences = review[0]
        for sentence in sentences:
            if len(sentence) > 1:
                raw_text = sentence[0].text.strip()
                text = tokenize(sentence[0].text.strip())[1]
                pos_set.append(tokenize(sentence[0].text.strip())[0])

                sample = [text, []]
                aspects = []

                # find the location of aspect terms without spaces
                # assumption : once tokenized, no additional term will be added but only the spaces will change
                opinions = sentence[1]
                for opinion in opinions:
                    start = int(opinion.attrib['from'])
                    end = int(opinion.attrib['to'])
                    if start != end:
                        t1 = raw_text[:start].count(" ")
                        t2 = raw_text[:end].count(" ")
                        aspects.append((start - t1, end - t2))

                curr = 0
                ai = 0
                wi = 0

                while wi < len(sample[0]):
                    if ai < len(aspects) and curr == aspects[ai][0]:
                        sample[1].append(1)
                        curr += len(sample[0][wi])
                        wi += 1
                        while curr < aspects[ai][1]:
                            sample[1].append(2)
                            curr += len(sample[0][wi])
                            wi += 1
                        ai += 1
                    else:
                        sample[1].append(0)
                        curr += len(sample[0][wi])
                        wi += 1

                assert len(sample[0])!=len(pos_set[-1]), "Mismatch between number of tokens and number of pos tags."

                dataset.append(sample)


    x = open(output + '_x.txt', 'w')
    y = open(output + '_y.txt', 'w')
    pos = open(output + '_pos.txt', 'w')

    for i in xrange(len(dataset) - 1):
        x.write(u' '.join(dataset[i][0]).encode('utf-8').strip() + '\n')
        y.write(' '.join(map(str, dataset[i][1])) + '\n')
        pos.write(' '.join(map(str, pos_set[i])) + '\n')

    # to remove the tailing \n
    x.write(' '.join(dataset[-1][0]))
    y.write(' '.join(map(str, dataset[-1][1])))
    pos.write(' '.join(map(str, pos_set[-1])))


def _add_words(file_path, word_set):
    with open(file_path) as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                word_set.add(token)


def generate_word2id(train_x, test_x):
    print "Generating word2id and id2word..."

    word_set = set()
    _add_words(train_x, word_set)
    _add_words(test_x, word_set)

    id2word = list(word_set)
    word2id = {word:id for id,word in enumerate(id2word)}
    print "word2id and id2word generated"
    return word2id, id2word


def save_word2id(train_x, test_x, word2id_path, id2word_path):
    word2id, id2word = generate_word2id(train_x, test_x)
    print "Saving word2id and id2word"
    with open(word2id_path, "wb") as f:
        pickle.dump(word2id, f)

    with open(id2word_path, "wb") as f:
        pickle.dump(id2word, f)
    print "Saving word2id and id2word completed"
    return word2id, id2word


def load_embeddings(train_x, test_x, embeddings_path, embeddings_out, save_dic = True):
    if save_dic:
        word2id, id2word = save_word2id(train_x, test_x,
                                        "/".join(embeddings_out.split("/")[:-1]) + "/word2id.pickle",
                                        "/".join(embeddings_out.split("/")[:-1]) + "/id2word.pickle")
    else:
        word2id, id2word = generate_word2id(train_x, test_x)
    print "Loading pretrained word embeddings"
    word_oov = np.ones(len(word2id), dtype=np.bool)
    found_count = 0

    with open(embeddings_path) as f:
        embedding_dim = len(f.readline().split()) - 1
        print "Embedding size = {0}".format(embedding_dim)
        embedding_matrix = np.ndarray((len(word2id), embedding_dim))
        f.seek(0)
        for line in f:
            line = line.split()
            word = line[0]
            if word in word2id:
                idx = word2id[word]
                found_count += 1
                # indicate word is not out of vocab
                word_oov[idx] = 0
                emb = map(float, line[1:])
                embedding_matrix[idx] = emb

    unk_emb = np.random.randn(embedding_dim)
    for i in xrange(len(word2id)):
        if word_oov[i]:
            embedding_matrix[i] = unk_emb

    np.save(embeddings_out, embedding_matrix)

    print "Embedding loading complete : {0} words unknown from {1} words".format(len(word2id) - found_count, len(word2id))

    return embedding_matrix


def generate_pos2id(train_pos, test_pos):
    print "generating pos2id and id2pos"
    pos_set = set()
    _add_words(train_pos, pos_set)
    _add_words(test_pos, pos_set)

    id2pos = list(pos_set)
    pos2id = {p:id for id,p in enumerate(id2pos)}
    print "generated pos2id and id2pos"
    return pos2id, id2pos


def save_pos2id(train_pos, test_pos, pos2id_path, id2pos_path):
    pos2id, id2pos = generate_pos2id(train_pos, test_pos)
    print "Saving pos2id and id2pos"
    with open(pos2id_path, "wb") as f:
        pickle.dump(pos2id, f)
    with open(id2pos_path, "wb") as f:
        pickle.dump(id2pos, f)
    print "Completed saving pos2id and id2pos"
    return pos2id, id2pos

if __name__ =="__main__":
    tokenize_data('data/ABSA15_Restaurants_Test.xml', 'data/restaurants_test')
    tokenize_data('data/ABSA-15_Restaurants_Train.xml', 'data/restaurants_train')

    train_x_path = "data/restaurants_train_x.txt"
    test_x_path = "data/restaurants_test_x.txt"
    train_pos_path = "data/restaurants_train_pos.txt"
    test_pos_path = "data/restaurants_test_pos.txt"

    embeddings_path = "./../Aspect Extraction Implementations/With-domain-specific-embeddings/Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction/data/embedding/gen.vec"
    embeddings_out = "data/embedding_matrix"

    load_embeddings(train_x_path, test_x_path, embeddings_path, embeddings_out, True)
    save_pos2id(train_pos_path, test_pos_path, "data/pos2id.pickle", "data/id2pos.pickle")