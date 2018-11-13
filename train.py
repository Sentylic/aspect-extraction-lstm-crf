import pickle
import torch
import numpy as np
from model import BiLSTM_CRF
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

word2id_path = "data/word2id.pickle"
id2word_path = "data/id2word.pickle"
pos2id_path = "data/pos2id.pickle"
id2pos_path = "data/id2pos.pickle"
embeddig_matrix_path = "data/embedding_matrix.npy"
train_x_path = "data/restaurants_train_x.txt"
train_y_path = "data/restaurants_train_y.txt"
test_x_path = "data/restaurants_test_x.txt"
test_y_path = "data/restaurants_test_y.txt"
train_pos_path = "data/restaurants_train_pos.txt"
test_pos_path = "data/restaurants_test_pos.txt"

with open(word2id_path) as f:
    word2id = pickle.load(f)

with open(id2word_path) as f:
    id2word = pickle.load(f)

with open(pos2id_path) as f:
    pos2id = pickle.load(f)

with open(id2pos_path) as f:
    id2pos = pickle.load(f)

embedding_matrix = np.load(embeddig_matrix_path)


def read_data(x_path, y_path):
    x = []
    y = []
    with open(x_path) as f:
        for line in f:
            x.append(torch.tensor(map(lambda x: word2id[x], line.split()), dtype=torch.long))

    with open(y_path) as f:
        for line in f:
            y.append(torch.tensor(map(int, line.split()), dtype=torch.long))

    return x,y


def read_pos(path):
    pos = []
    with open(path) as f:
        for line in f:
            pos.append(torch.tensor(map(lambda x: pos2id[x], line.split()), dtype=torch.long))
    return pos

train_x, train_y = read_data(train_x_path, train_y_path)
test_x, test_y = read_data(test_x_path, test_y_path)

train_pos = read_pos(train_pos_path)
test_pos = read_pos(test_pos_path)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag2id = {"0": 0, "1": 1, "2": 2, START_TAG: 3, STOP_TAG: 4}

EMBEDDING_DIM = 300
HIDDEN_DIM = 150
POS_EMBEDDING_DIM = 100


# model = torch.load("saved_models/best_model_lstm_crf_pos.pt")
best_f = 0
model = BiLSTM_CRF(len(word2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM, len(pos2id), POS_EMBEDDING_DIM)
model.word_embeds.weight = nn.Parameter(torch.Tensor(embedding_matrix))
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

scores = []

count = 0
for epoch in range(50):
    print("\nepoch : {0}".format(epoch + 1))
    for x, x_pos, y in tqdm(zip(train_x, train_pos, train_y)):
        count += 1
        model.zero_grad()

        sentence_in = x
        sentence_pos = x_pos
        targets = y

        assert len(sentence_pos)==len(sentence_in), count
        assert len(sentence_pos)==len(targets)

        loss = model.neg_log_likelihood(sentence_in, sentence_pos, targets)

        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        precision = 0
        recall = 0
        s = 0
        g = 0
        s_g = 0

        for x, x_pos, y in zip(test_x, test_pos, test_y):
            sentence_in = x
            sentence_pos = x_pos
            _, prediction = model(sentence_in, sentence_pos)

            prediction = torch.LongTensor(prediction)
            y = torch.LongTensor(map(int, y))

            prediction = prediction.tolist()
            y = y.tolist()

            i = 0
            while (i < len(prediction)):
                if prediction[i] == 1:
                    s += 1
                    if y[i] == 1:
                        g += 1
                        i += 1
                        if i >= len(prediction):
                            s_g += 1
                            continue
                        while (i < len(prediction) and prediction[i] == 2):
                            if not y[i] == 2:
                                i += 1
                                break
                            i += 1
                        else:
                            s_g += 1
                    else:
                        i += 1
                elif y[i] == 1:
                    g += 1
                    i += 1
                else:
                    i += 1

        precision += float(s_g) / s
        recall += float(s_g) / g
        print(precision, recall)
        #             if len(y) == (prediction==y).sum().tolist():
        #                 correct += 1

        f_score = 2 * precision * recall / (precision + recall)
        scores.append(f_score)

        if best_f < f_score:
            best_f = f_score
            # save the model
            print("Saving the best model up to now")
            torch.save(model, "saved_models/best_model_lstm_crf_pos.pt")

        print "f_score %f\n" % f_score
