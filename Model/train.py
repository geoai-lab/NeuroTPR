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
Model_DIR = '/home/jiminwan/NeuroTPR_project/Model'


def tag_dataset(dataset, model_path):
    model2 = load_keras_model(model_path)
    print("successully load model!")

    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        token, char, labels, ner, pos, strings = data
        tokens = np.asarray([token])
        ners = np.asarray([ner])
        chars = np.asarray([char])
        poss = np.asarray([pos])
        texts = np.asarray([strings])
        pred = model2.predict([tokens, chars, ners, poss, texts], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i + 1)

    return predLabels, correctLabels


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

    trainSentences1 = readfile("/home/jiminwan/NeuroTPR_project/Model/data/WikiTPR1000_train_add_features.txt")
    trainSentences2 = readfile("/home/jiminwan/NeuroTPR_project/Model/data/wnut17train_add_features.txt")
    testSentences = readfile("/home/jiminwan/NeuroTPR_project/Model/data/Harvey1000.txt")
    print("Finishing reading the training data from file! ")

    trainSentences1 = addCharInformatioin(trainSentences1)
    trainSentences2 = addCharInformatioin(trainSentences2)
    print("Finishing adding the character information! ")

    labelSet = set()
    posSet = set()
    all_words = {}

    for sentence in trainSentences1:
        for token, char, label, ner, pos in sentence:
            labelSet.add(label)
            posSet.add(pos)
            all_words[token.lower()] = True

    for sentence in trainSentences2:
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
    # print(char2Idx_caseless)
    ner2Idx = get_ner_embedding()

    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    pos2Idx = {}
    for pos in posSet:
        pos2Idx[pos] = len(pos2Idx)

    pos2Idx["UNKNOWN"] = len(pos2Idx)

    nerEmbeddings = np.identity(len(ner2Idx), dtype='float32')
    posEmbeddings = np.identity(len(pos2Idx), dtype='float32')

    idx2Label = {v: k for k, v in label2Idx.items()}
    print(idx2Label)

    # -------------- Process the word embedding ------------ #
    print("Begin to add the word embedding information! ")
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("/home/jiminwan/NeuroTPR_project/embedding/glove.twitter.27B.200d.txt", encoding="utf-8")

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
    np.save("outputs/ner2Idx.npy", ner2Idx)
    np.save("outputs/char2Idx_caseless.npy", char2Idx_caseless)
    np.save("outputs/pos2Idx.npy", pos2Idx)

    # ----------------------------------------#

    train_set1 = padding(createMatrices_char(trainSentences1, word2Idx, label2Idx, char2Idx, char2Idx_caseless, ner2Idx, pos2Idx))
    train_set2 = padding(createMatrices_char(trainSentences2, word2Idx, label2Idx, char2Idx, char2Idx_caseless, ner2Idx, pos2Idx))
    # test_set = padding(createMatrices_char(testSentences, word2Idx, label2Idx, char2Idx, ner2Idx, pos2Idx))


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
    # output = concatenate([words, char_lstm1, words_elmo], axis=-1)

    output_lstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

    my_crf = CRF(len(label2Idx), sparse_target=True, name='CRF_layer')(output_lstm)

    model = Model(inputs=[words_input, character_input1, character_input2, ner_input, pos_input, words_elmo_input], outputs=[my_crf])
    # model = Model(inputs=[words_input, character_input1, character_input2, words_elmo_input], outputs=[my_crf])

    # my_optimizer = optimizers.SGD(loss='sparse_categorical_crossentropy', optimizer='sgd', lr=0.15, decay=0.99)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

    model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    # model.summary()

    # -----------------  Model Built Up Finished -------------------- #

    # ----------------- K-fold Training and validation ----------------- #

    # Training and validation dataset
    train_batch1, train_batch_len1 = createBatches(train_set1)
    train_batch2, train_batch_len2 = createBatches(train_set2)


    for epoch in range(epochs):
        train_accuracy = 0.0
        print("Epoch %d/%d" % (epoch, epochs))
        a = Progbar(len(train_batch_len1))
        for i, batch in enumerate(iterate_minibatches_char(train_batch1, train_batch_len1)):
            labels, tokens, chars, chars2, ners, poss, strings = batch
            results = model.train_on_batch([tokens, chars, chars2, ners, poss, strings], labels)
            train_accuracy += results[1]
            a.update(i)
        a.update(i + 1)
        print(train_accuracy/(i+1))
        if train_accuracy/(i+1) > 0.98:
            break

    # -------------- Save moddel -----------------#
    # save_keras_model(model, Model_DIR, name1="/outputs/NeuroTPR1.json", name2="/outputs/NeuroTPR1.h5")

    for epoch in range(epochs):
        train_accuracy = 0.0
        print("Epoch %d/%d" % (epoch, epochs))
        a = Progbar(len(train_batch_len2))
        for i, batch in enumerate(iterate_minibatches_char(train_batch2, train_batch_len2)):
            labels, tokens, chars, chars2, ners, poss, strings = batch
            results = model.train_on_batch([tokens, chars, chars2, ners, poss, strings], labels)
            train_accuracy += results[1]
            a.update(i)
        a.update(i + 1)
        print(train_accuracy/(i+1))
        if train_accuracy/(i+1) > 0.99:
            break

    # -------------- Save moddel -----------------#
    save_keras_model(model, Model_DIR, name1="/outputs/NeuroTPR2.json", name2="/outputs/NeuroTPR2.h5")


    #   Performance on test dataset
    # print("Test the trained model on the testing dataset:")
    # predLabels, correctLabels = tag_dataset(test_batch, Model_DIR)
    # pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
    # print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))