import numpy as np
import copy
import json
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from emoji import get_emoji_regexp
from ELMo import ElmoEmbeddingLayer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


model_path = "outputs/"
wordembedding_path = "outputs/word2Idx.npy"
labelset_path = "outputs/idx2Label.npy"
charembedding_path = "outputs/char2Idx.npy"
charembedding_path2 = "outputs/char2Idx_caseless.npy"
posembedding_path = "outputs/pos2Idx.npy"


def load_keras_model(modelDIR):
    json_file = open(modelDIR + 'NeuroTPR.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
        'CRF': CRF, 'crf_loss': crf_loss, "crf_viterbi_accuracy": crf_viterbi_accuracy,
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer})
    loaded_model.load_weights(modelDIR + 'NeuroTPR.h5')
    print("Loaded NeuroTPR model from disk")
    return loaded_model


def delete_emoji(raw_text):
    return get_emoji_regexp().sub(r'', raw_text)


def tweet_pos_tagger(text_list):
    word_idx = []
    start_idx = 1
    for word in text_list:
        word_idx.append(start_idx)
        start_idx += len(word)+1

    tokenized_text = ["URL" if word.startswith("http") else word for word in text_list]
    pos_taggers = pos_tag(tokenized_text)
    pos_sequence = [item[1] for item in pos_taggers]

    return tokenized_text, word_idx, pos_sequence


def preprocess_tweet(text):
    text = delete_emoji(text)
    tweet_token = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
    results = tweet_token.tokenize(text)
    results = [item[1:] if (item.startswith("#") and len(item)>1) else item for item in results]

    tokens, indexes, poss = tweet_pos_tagger(results)
    new_tweets = []
    for i in range(0, len(tokens)):
        new_tweets.append([tokens[i], indexes[i], poss[i]])

    return new_tweets


def addCharInformatioin(Sentence):
    for i, data in enumerate(Sentence):
            chars = [c for c in data[0]]
            Sentence[i] = [data[0], chars, data[1], data[2]]
    return Sentence


def createMatrices_nolabel_char(sentence, word2Idx, char2Idx, char2Idx_caseless, pos2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    all_features = []

    wordCount = 0
    unknownWordCount = 0

    wordIndices = []
    charIndices1 = []
    charIndices2 = []
    posIndices = []
    wordStrings = ""

    for word, char, index, pos in sentence:
        wordCount += 1
        wordStrings = wordStrings + word + " "

        if word in word2Idx:
            wordIdx = word2Idx[word]
        elif word.lower() in word2Idx:
            wordIdx = word2Idx[word.lower()]
        else:
            wordIdx = unknownIdx
            unknownWordCount += 1
        charIdx1 = []
        charIdx2 = []
        for x in char:
            if x in char2Idx:
                charIdx1.append(char2Idx[x])
                charIdx2.append(char2Idx_caseless[x])
            else:
                charIdx1.append(char2Idx['UNKNOWN'])
                charIdx2.append(char2Idx_caseless['UNKNOWN'])

        # Get the label and map to int
        wordIndices.append(wordIdx)
        charIndices1.append(charIdx1)
        charIndices2.append(charIdx2)

        if pos in pos2Idx:
            posIndices.append(pos2Idx[pos])
        else:
            posIndices.append(pos2Idx['UNKNOWN'])

    all_features.append([wordIndices, charIndices1, charIndices2, posIndices, wordStrings[:-1]])

    return all_features


def padding(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[1]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1], 52, padding='post')
        Sentences[i][2] = pad_sequences(Sentences[i][2], 52, padding='post')
    return Sentences


def build_senteceMatrix(sentence):

    features = []
    wordIndices = []
    indexIndices = []
    for word, label, pos in sentence:
        # Get the label and map to int
        wordIndices.append(word)
        indexIndices.append(int(label))
    features.append([wordIndices, indexIndices])

    return features


def geoparse_tweet(tweet_features, tweet_raw, model, iinx, binx):
    toponyms_set = []

    for i, sentence in enumerate(tweet_features):
        tokens, chars, chars2, poss, text = sentence
        tokens = np.asarray([tokens])
        chars = np.asarray([chars])
        chars2 = np.asarray([chars2])
        poss = np.asarray([poss])
        text = np.asarray([text])

        if len(text[0]) < 5:
            print("Too short input for the model")
            break

        pred = model.predict([tokens, chars, chars2, poss, text], verbose=False)[0]  #

        pred = pred.argmax(axis=-1)  # Predict the classes

        tokenIdx = 0
        toponym_item = {}
        while tokenIdx < len(pred):
            toponym = ""
            if pred[tokenIdx] == binx:  # A new chunk starts
                old_idx = tokenIdx
                toponym = tweet_raw[i][0][tokenIdx]
                tokenIdx += 1

                while tokenIdx < len(pred) and pred[tokenIdx] == iinx:
                    toponym += (" " + tweet_raw[i][0][tokenIdx])
                    tokenIdx += 1

                if toponym.endswith(","):
                    toponym = toponym[:-2]

                if toponym.find("Harvey") == -1 and len(toponym) > 1:
                    toponym_item["location_name"] = toponym
                    toponym_item["start_idx"] = tweet_raw[i][1][old_idx]
                    toponym_item["end_idx"] = tweet_raw[i][1][old_idx] + len(toponym) - 1
                    toponyms_set.append(dict(toponym_item))

            elif pred[tokenIdx] == iinx:
                tokenIdx += 1

            else:
                tokenIdx += 1

    return toponyms_set


def geoparse(tweet, model_path, wordembedding_path, labelset_path, charembedding_path, charembedding_path2,
             posembedding_path):

    print("Load the NeuroTPR model weights")
    model = load_keras_model(model_path)

    print("Load the word and linguistic embeddings from npy files")
    word2Idx = np.load(wordembedding_path, allow_pickle=True).item()
    labelIdx = np.load(labelset_path, allow_pickle=True).item()
    char2Idx = np.load(charembedding_path, allow_pickle=True).item()
    char2Idx_caseless = np.load(charembedding_path2, allow_pickle=True).item()
    pos2Idx = np.load(posembedding_path, allow_pickle=True).item()
    inv_labelIdx = {v: k for k, v in labelIdx.items()}

    bloc_index = inv_labelIdx["B-location"]
    iloc_index = inv_labelIdx["I-location"]

    processed_tweet = preprocess_tweet(tweet)

    processed_tweet_bak = copy.deepcopy(processed_tweet)

    processed_tweet_char = addCharInformatioin(processed_tweet)

    tweet_all_features = padding(createMatrices_nolabel_char(processed_tweet_char, word2Idx, char2Idx,
                                                              char2Idx_caseless, pos2Idx))
    tweet_word_index = build_senteceMatrix(processed_tweet_bak)

    result_json = geoparse_tweet(tweet_all_features, tweet_word_index, model, iloc_index, bloc_index)
    result_json = json.dumps(result_json)

    print("The Toponym Recognition results are:")
    print(result_json)
    print("Toponym Recognition step is over")


my_tweet = "The City of Dallas has now opened Shelter #3 at Samuel Grand Recreation Center, 6200 E. Grand Ave. #HurricaneHarvey"
print("Input tweet content is:")
print(my_tweet)
print("Processing on the input texts now ...")
geoparse(my_tweet, model_path, wordembedding_path, labelset_path, charembedding_path, charembedding_path2,
         posembedding_path)