import numpy as np
from keras.models import Model, model_from_json
from keras.layers import TimeDistributed, Embedding, Input, LSTM, Bidirectional, concatenate
from preprocess import readfile, createBatches, createMatrices_char, iterate_minibatches_char, addCharInformatioin,\
    padding
from keras.utils import Progbar
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from validation import compute_f1
from ELMo import ElmoEmbeddingLayer

epochs = 50
Model_DIR = '...'

def get_ner_embedding():
    ner2Idx = {'LOCATION': 0, 'ORGANIZATION': 1, 'PERSON': 2, 'O': 3, 'PADDING_TOKEN': 4}
    return ner2Idx


def get_char_embedding():
    char2Idx = {"PADDING": 0, "UNKNOWN": 1}
    char2Idx_caseless = {"PADDING": 0, "UNKNOWN": 1}

    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    for c in " 0123456789abcdefghijklmnopqrstuvwxyz.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx_caseless[c] = len(char2Idx_caseless)

    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        char2Idx_caseless[c] = char2Idx_caseless[c.lower()]

    return char2Idx, char2Idx_caseless


def save_keras_model(model, modelDIR, name1="/outputs/NeuroTPR.json", name2="/outputs/NeuroTPR.h5"):
    model_json = model.to_json()
    with open(modelDIR + name1, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(modelDIR + name2)
    print("Saved model to disk")


def load_keras_model(modelDIR):
    json_file = open(modelDIR+'/outputs/NeuroTPR.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
                'CRF': CRF, 'crf_loss': crf_loss, "crf_viterbi_accuracy": crf_viterbi_accuracy,
                'ElmoEmbeddingLayer': ElmoEmbeddingLayer})
    loaded_model.load_weights(modelDIR+'/outputs/NeuroTPR.h5')
    print("Loaded model from disk")
    return loaded_model


if __name__ == '__main__':

    trainSentences = readfile("...")
    testSentences = readfile("...")
    print("Finishing reading the training data from file! ")

    trainSentences = addCharInformatioin(trainSentences)
    print("Finishing adding the character information! ")

    labelSet = set()
    posSet = set()
    all_words = {}

    for sentence in trainSentences1:
        for token, char, label, ner, pos in sentence:
            labelSet.add(label)
            posSet.add(pos)
            all_words[token.lower()] = True

    for sentence in testSentences:
        for token, label, ner, pos in sentence:
            all_words[token.lower()] = True

    print("Finishing cleaning the raw text! ")

    # -------------- Process Other linguistic embedding ------------ #
    char2Idx, char2Idx_caseless = get_char_embedding()

    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    pos2Idx = {}
    for pos in posSet:
        pos2Idx[pos] = len(pos2Idx)

    pos2Idx["UNKNOWN"] = len(pos2Idx)

    posEmbeddings = np.identity(len(pos2Idx), dtype='float32')

    idx2Label = {v: k for k, v in label2Idx.items()}

    # -------------- Process the word embedding ------------ #
    print("Begin to add the word embedding information! ")
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("../embedding/glove.twitter.27B.200d.txt", encoding="utf-8")

    for line in fEmbeddings:
        split = line.strip().split(" ")

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if split[0].lower() in all_words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)

    wordEmbeddings = np.array(wordEmbeddings)
    print("Finishing constructing word-vectors dictionary! ")

    np.save("outputs/idx2Label.npy", idx2Label)
    np.save("outputs/word2Idx.npy", word2Idx)
    np.save("outputs/char2Idx.npy", char2Idx)
    np.save("outputs/char2Idx_caseless.npy", char2Idx_caseless)
    np.save("outputs/pos2Idx.npy", pos2Idx)

    # ----------------------------------------#

    train_set = padding(createMatrices_char(trainSentences, word2Idx, label2Idx, char2Idx, char2Idx_caseless, pos2Idx))


    # ------------- Create the char-word-ELMo-BiLSTM-CRF toponym recognition model -----------------#

    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                      trainable=False)(words_input)

    ner_input = Input(shape=(None,), dtype='int32', name='ner_input')
    ner = Embedding(input_dim=nerEmbeddings.shape[0], output_dim=nerEmbeddings.shape[1], weights=[nerEmbeddings],
                       trainable=False)(ner_input)

    pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
    pos = Embedding(input_dim=posEmbeddings.shape[0], output_dim=posEmbeddings.shape[1], weights=[posEmbeddings],
                       trainable=False)(pos_input)

    character_input1 = Input(shape=(None, 52), name='char_input_normal')
    embed_char_out1 = TimeDistributed(Embedding(input_dim=len(char2Idx), output_dim=50, embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding1')(character_input1)
    char_lstm1 = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False, recurrent_dropout=0.25)),
                                name='char_LSTM1')(embed_char_out1)

    character_input2 = Input(shape=(None, 52), name='char_input_caseless')
    embed_char_out2 = TimeDistributed(Embedding(input_dim=len(char2Idx_caseless)-26, output_dim=50, embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding2')(character_input2)
    char_lstm2 = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False, recurrent_dropout=0.25)),
                                name='char_LSTM2')(embed_char_out2)

    words_elmo_input = Input(shape=(1,),  dtype='string', name='words_elmo_input')

    words_elmo = ElmoEmbeddingLayer(trainable=False)(words_elmo_input)

    output = concatenate([words, ner, pos, char_lstm1, char_lstm2, words_elmo], axis=-1)

    output_lstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

    my_crf = CRF(len(label2Idx), sparse_target=True, name='CRF_layer')(output_lstm)

    model = Model(inputs=[words_input, character_input1, character_input2, ner_input, pos_input, words_elmo_input], outputs=[my_crf])

    model.compile(optimizer='nadam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    # model.summary()

    # -----------------  Model Built Up Finished -------------------- #

    # ----------------- K-fold Training and validation ----------------- #

    # Training and validation dataset
    train_batch, train_batch_len = createBatches(train_set)


    for epoch in range(epochs):
        train_accuracy = 0.0
        print("Epoch %d/%d" % (epoch, epochs))
        a = Progbar(len(train_batch_len))
        for i, batch in enumerate(iterate_minibatches_char(train_batch, train_batch_len)):
            labels, tokens, chars, chars2, poss, strings = batch
            results = model.train_on_batch([tokens, chars, chars2, poss, strings], labels)
            train_accuracy += results[1]
            a.update(i)
        a.update(i + 1)
        print(train_accuracy/(i+1))
        if train_accuracy/(i+1) > 0.98:
            break

    # -------------- Save moddel -----------------#
    save_keras_model(model, Model_DIR, name1="/outputs/NeuroTPR.json", name2="/outputs/NeuroTPR.h5")
