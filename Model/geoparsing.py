import numpy as np
import copy
import os
from re import finditer
from ELMo import ElmoEmbeddingLayer
from keras.models import model_from_json
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from prediction_pre import readfile_nolabel, createMatrices_nolabel_char, build_senteceMatrix, padding, \
    addCharInformatioin

model_path = "/home/jiminwan/NeuroTPR_project/Model/outputs2/"
OUTPUT_DIR = "/home/jiminwan/NeuroTPR_project/Model/data/testing"
CORPUS_DIR = "/home/jiminwan/NeuroTPR_project/Model/data/testing/HarveyTweet"
wordembedding_path = "/home/jiminwan/NeuroTPR_project/Model/outputs2/word2Idx.npy"
labelset_path = "/home/jiminwan/NeuroTPR_project/Model/outputs2/idx2Label.npy"
charembedding_path = "/home/jiminwan/NeuroTPR_project/Model/outputs2/char2Idx.npy"
charembedding_path2 = "/home/jiminwan/NeuroTPR_project/Model/outputs2/char2Idx_caseless.npy"
posembedding_path = "/home/jiminwan/NeuroTPR_project/Model/outputs2/pos2Idx.npy"


def load_keras_model(modelDIR):
    json_file = open(modelDIR + 'NeuroTPR2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
        'CRF': CRF, 'crf_loss': crf_loss, "crf_viterbi_accuracy": crf_viterbi_accuracy,
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer})
    loaded_model.load_weights(modelDIR + 'NeuroTPR2.h5')
    print("Loaded model from disk")
    return loaded_model


def street_detector(texts):
    results = {}
    stree_pattern = r'I - ([0-9]+)'
    for match in finditer(stree_pattern, texts):
        results[match.group()] = match.span()

    return results


def tag_article(input, test_article, model, iinx, binx):
    ## per article conducting prediction

    result_line = ""

    for i, sentence in enumerate(input):
        ## per sentence as the input
        tokens, chars, chars2, ners, poss, texts = sentence
        tokens = np.asarray([tokens])
        chars = np.asarray([chars])
        chars2 = np.asarray([chars2])
        poss = np.asarray([poss])
        texts = np.asarray([texts])

        if len(texts[0]) <= 1:
            break

        # start_time = time.time()
        pred = model.predict([tokens, chars, ners, poss, texts], verbose=False)[0]
        # end_time = time.time()
        # print("Takes {0} seconds".format(end_time - start_time))

        pred = pred.argmax(axis=-1)  # Predict the classes

        # print(pred)
        tokenIdx = 0

        while tokenIdx < len(pred):
            toponym = ""
            if pred[tokenIdx] == binx:  # A new chunk starts
                old_idx = tokenIdx
                toponym = test_article[i][0][tokenIdx]
                tokenIdx += 1

                while tokenIdx < len(pred) and pred[tokenIdx] == iinx:
                    toponym += (" " + test_article[i][0][tokenIdx])
                    tokenIdx += 1

                if toponym.endswith(","):
                    print(toponym)
                    toponym = toponym[:-2]

                if len(toponym) > 1:
                    result_line += toponym + ",," + toponym + ",,30,,140,,"
                    result_line += str(test_article[i][1][old_idx]) + ",," + str(
                        test_article[i][1][old_idx] + len(toponym) - 1) + "||"

            elif pred[tokenIdx] == iinx:
                toponym = test_article[i][0][tokenIdx]
                print(toponym)
                tokenIdx += 1

            else:
                tokenIdx += 1

    return result_line


def tag_corpus(corpus_path, corpus_name, word2Idx, idx2Label, char2Idx, char2Idx_caseless, pos2Idx):
    output_path = os.path.join(OUTPUT_DIR, corpus_name + ".txt")
    fwriter = open(output_path, "a+")
    files = os.listdir(corpus_path)

    bloc_index = idx2Label["B-location"]
    iloc_index = idx2Label["I-location"]

    model = load_keras_model(model_path)

    for fileid in range(0, len(files), 1):
        article_path = os.path.join(CORPUS_DIR, str(fileid) + ".txt")

        sentences = readfile_nolabel(article_path)

        new_sentences = copy.deepcopy(sentences)

        sentences_char = addCharInformatioin(sentences)

        processed_sentences = padding(createMatrices_nolabel_char(sentences_char, word2Idx, char2Idx,
                                                                  char2Idx_caseless, pos2Idx))

        sentences = build_senteceMatrix(new_sentences)

        # processed_sentences, sentences_length = createBatches(processed_sentences)

        ## open the file writter

        # ## process one article
        temp = tag_article(processed_sentences, sentences, model, iloc_index, bloc_index)

        print("processing on {0} article".format(fileid))
        # print(temp)

        fwriter.writelines(temp + '\n')

    fwriter.close()


word2Idx = np.load(wordembedding_path, allow_pickle=True).item()

labelIdx = np.load(labelset_path, allow_pickle=True).item()

char2Idx = np.load(charembedding_path, allow_pickle=True).item()

char2Idx_caseless = np.load(charembedding_path, allow_pickle=True).item()

pos2Idx = np.load(posembedding_path, allow_pickle=True).item()

inv_labelIdx = {v: k for k, v in labelIdx.items()}

tag_corpus(CORPUS_DIR, "HarveyTweet", word2Idx, inv_labelIdx, char2Idx, char2Idx_caseless, pos2Idx)
