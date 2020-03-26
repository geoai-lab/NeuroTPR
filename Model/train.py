import numpy as np
from preprocess import readfile, createBatches, createMatrices_char, iterate_minibatches_char, addCharInformatioin,\
    padding
import keras.backend as K
from keras.utils import Progbar
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from NeuroTPR import NeuroTPR_initiate
from DM_NLP import DMNLP_initiate


epochs1 = 35
epochs2 = 20
embedding_choice = 0 # 0: Twitter_Glove; 1: Normal_Glove
Model_DIR = 'Change with your own Root Directory'


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


if __name__ == '__main__':
	# Read training file as txt
    trainSentences1 = readfile("../data/WikiTPR3000_train_add_features.txt")
    trainSentences2 = readfile("../data/wnut17train2_add_features.txt")
    testSentences = readfile("../data/Harvey1000.txt")
    print("Finishing reading the training data from file! ")

    # Process the character features from words
    trainSentences1 = addCharInformatioin(trainSentences1)
    trainSentences2 = addCharInformatioin(trainSentences2)
    print("Finishing adding the character information!!!")

    labelSet = set()
    posSet = set()
    all_words = {}

    for TrainSentences in [trainSentences1, trainSentences2]:
    	for sentence in TrainSentences:
	        for token, char, label, ner, pos in sentence:
	            labelSet.add(label)
	            posSet.add(pos)
	            all_words[token.lower()] = True

    for sentence in testSentences:
        for token, label, ner, pos in sentence:
            all_words[token.lower()] = True

    print("Finishing cleaning the raw text! ")

    # -------------- Process the linguistic embedding ------------ #
    # Create character feature lookup dictionary
    char2Idx, char2Idx_caseless = get_char_embedding()

    # Create a mapping for the labels
    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    idx2Label = {v: k for k, v in label2Idx.items()}
    print(idx2Label)

    # Create pos feature lookup dictionary
    pos2Idx = {}
    for pos in posSet:
        pos2Idx[pos] = len(pos2Idx)

    pos2Idx["UNKNOWN"] = len(pos2Idx)
    posEmbeddings = np.identity(len(pos2Idx), dtype='float32')

    # Create ner feature lookup dictionary
    ner2Idx = get_ner_embedding()
    nerEmbeddings = np.identity(len(ner2Idx), dtype='float32')

    # -------------- Process the word embedding ------------ #
    print("Begin to add the word embedding information! ")
    word2Idx = {}
    wordEmbeddings = []

    if embedding_choice == 0:
        fEmbeddings = open("../embedding/glove.twitter.27B.200d.txt", encoding="utf-8")
    else:
        fEmbeddings = open("../embedding/glove.6B.300d.txt", encoding="utf-8")

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

    # Save all embeddings and linguistic features lookup into files
    np.save("outputs/idx2Label.npy", idx2Label)
    np.save("outputs/word2Idx.npy", word2Idx)
    np.save("outputs/char2Idx.npy", char2Idx)
    np.save("outputs/ner2Idx.npy", ner2Idx)
    np.save("outputs/char2Idx_caseless.npy", char2Idx_caseless)
    np.save("outputs/pos2Idx.npy", pos2Idx)

    # ----------------------------------------#

    # Convert training senetences into Keras-compatitable format
    train_set1 = padding(createMatrices_char(trainSentences1, word2Idx, label2Idx, char2Idx, char2Idx_caseless, pos2Idx))
    train_set2 = padding(createMatrices_char(trainSentences2, word2Idx, label2Idx, char2Idx, char2Idx_caseless, pos2Idx))

    # ------------- Build up NeuroTPR model architecture -----------------#

    p = {"label_length": len(idx2Label), "char_length": len(char2Idx), "char_length_caseless": len(char2Idx)-26}
    model = NeuroTPR_initiate(wordEmbeddings, posEmbeddings, params=p)

    model.compile(optimizer='nadam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.summary()

    # -----------------  Model Built Up Finished -------------------- #

    # ----------------- Model Training and Weights Saving ----------------- #

    # Training senetences to batches

    train_batch1, train_batch_len1 = createBatches(train_set1)
    train_batch2, train_batch_len2 = createBatches(train_set2)

    for epoch in range(epochs1):
        train_accuracy = 0.0
        print("Epoch %d/%d" % (epoch, epochs))
        a = Progbar(len(train_batch_len1))
        for i, batch in enumerate(iterate_minibatches_char(train_batch1, train_batch_len1)):
            labels, tokens, chars, chars2, poss, strings = batch
            results = model.train_on_batch([tokens, chars, chars2, poss, strings], labels)
            train_accuracy += results[1]
            a.update(i)
        a.update(i + 1)
        print(train_accuracy/(i+1))
        if train_accuracy/(i+1) > 0.99:
            break

    for epoch in range(epochs2):
        train_accuracy = 0.0
        print("Epoch %d/%d" % (epoch, epochs))
        a = Progbar(len(train_batch_len2))
        for i, batch in enumerate(iterate_minibatches_char(train_batch2, train_batch_len2)):
            labels, tokens, chars, chars2, poss, strings = batch
            results = model.train_on_batch([tokens, chars, chars2, poss, strings], labels) # 
            train_accuracy += results[1]
            a.update(i)
        a.update(i + 1)
        print(train_accuracy/(i+1))

    # -------------- Save moddel -----------------#

    model_name_j = "/outputs/NeuroTPR.json"
    model_name_h = "/outputs/NeuroTPR.h5"

    save_keras_model(model, Model_DIR, name1=model_name_j, name2=model_name_h)
