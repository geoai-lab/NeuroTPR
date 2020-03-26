import nltk
from nltk.tag import StanfordNERTagger
from nltk import RegexpParser
from nltk import conlltags2tree, tree2conlltags

# nltk.download('averaged_perceptron_tagger')


def sentence_to_file(filename="../trainingdata/temp.txt", psentence=[]):
    with open(filename, "a") as wf:
        for item in psentence:
            wf.write(item[0]+"\t"+item[1]+"\n")

        wf.write("\n")
        wf.close()


def tweet_ner_tagger(text_list, st, cp):
    text = ["URL" if word[0].startswith("http") else word[0] for word in text_list]
    gold_tag = [word[1] for word in text_list]

    tokenized_text = text

    ner_taggers = st.tag(tokenized_text)
    pos_taggers = nltk.pos_tag(tokenized_text)
    chunk_taggers = tree2conlltags(cp.parse(pos_taggers))

    ner_sequence = [item[1] for item in ner_taggers]
    pos_sequence = [item[1] for item in pos_taggers]
    chunking_sequence = [item[1] for item in chunk_taggers]

    return text, gold_tag, ner_sequence, pos_sequence, chunking_sequence


def iterate_file(filename_read, filename_write, st, cp):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    read_f = open(filename_read, "r+")
    write_f = open(filename_write, "w+")
    j = 0
    sentence = []
    for line in read_f:
        line = line.rstrip()
        if len(line) == 0:
            # if len(sentence) > 0 and len(sentence) < 25:
            # Here to add the linguistics features
            c1, c2, c3, c4, c5 = tweet_ner_tagger(sentence, st, cp)
            if len(c1) != len(c3):
                sentence_to_file(psentence=sentence)
                sentence = []
                print("drop!!!!")
                continue
            for i in range(0, len(c1)):
                write_f.write(c1[i] + "\t" + c2[i] + "\t" + c3[i] + "\t" + c4[i] + "\t" + c5[i] + "\n")
            write_f.write("\n")

            sentence = []
            j += 1
            print(j)
            continue

        splits = line.split('\t')
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        c1, c2, c3, c4, c5 = tweet_ner_tagger(sentence, st)
        for i in range(0, len(c1)):
            write_f.write(c1[i] + "\t" + c2[i] + "\t" + c3[i] + "\t" + c4[i] + "\t" + c5[i] + "\n")
        write_f.write("\n")
        sentence = []

    write_f.close()
    read_f.close()
    print("OVER!!!")


if __name__ == "__main__":
    raw_dataset_path = "../trainingdata/wnut17train2.txt"
    processed_dataset_path = "../trainingdata/wnut17train2_add_features.txt"



    stanford_tagger = StanfordNERTagger('../stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                                        '../stanford-ner/stanford-ner.jar',
                                        encoding='utf-8')

    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = RegexpParser(grammar)

    iterate_file(raw_dataset_path, processed_dataset_path, stanford_tagger, cp)




