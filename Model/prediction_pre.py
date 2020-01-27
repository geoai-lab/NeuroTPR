from keras.preprocessing.sequence import pad_sequences

def build_senteceMatrix(sentences):

    dataset = []
    for sentence in sentences:
        wordIndices = []
        indexIndices = []
        for word, label, ner, pos in sentence:
            # Get the label and map to int
            wordIndices.append(word)
            indexIndices.append(int(label))
        dataset.append([wordIndices, indexIndices])

    return dataset

def readfile_nolabel(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    current_index = 1
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
        sentence.append([splits[0], current_index, splits[1], splits[2]])
        current_index += (1 + len(splits[0]))

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []

    return sentences


def createMatrices_nolabel_char(sentences, word2Idx, char2Idx, char2Idx_caseless, pos2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        charIndices1 = []
        charIndices2 = []
        posIndices = []
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

        dataset.append([wordIndices, charIndices1, charIndices2, posIndices, wordStrings[:-1]])

    return dataset


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
