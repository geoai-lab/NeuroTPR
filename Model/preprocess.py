import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        line = line.rstrip()
        if len(line) == 0:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        if len(splits[0]) == 0:
            continue
        sentence.append([splits[0], splits[1], splits[2], splits[3]])


    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = list(set(l))
    random.shuffle(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def addCharInformatioin(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1], data[2], data[3]]
    return Sentences


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


def createMatrices_char(sentences, word2Idx, label2Idx, char2Idx, char2Idx_caseless, pos2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        charIndices = []
        charIndices2 = []
        labelIndices = []
        posIndices = []
        # nerIndices = []
        wordStrings = ""

        for word, char, label, ner, pos in sentence:
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
                if x not in char2Idx:
                    x = "UNKNOWN"
                charIdx1.append(char2Idx[x])
                charIdx2.append(char2Idx_caseless[x])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            charIndices.append(charIdx1)
            charIndices2.append(charIdx2)
            labelIndices.append(label2Idx[label])
            posIndices.append(pos2Idx[pos])
            # nerIndices.append(ner2Idx[ner])

        dataset.append([wordIndices, charIndices, charIndices2, labelIndices, posIndices, wordStrings[:-1]])

    return dataset


def iterate_minibatches_char(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        char = []
        char2 = []
        labels = []
        words = []
        poss = []
        # ners = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, ch, ch2, l, p, word = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            char.append(ch)
            char2.append(ch2)
            labels.append(l)
            words.append(word)
            poss.append(p)
            # ners.append(n)

        yield np.asarray(labels), np.asarray(tokens), np.asarray(char), np.asarray(char2),\
            np.asarray(poss), np.array(words, dtype=object)[:, np.newaxis]

