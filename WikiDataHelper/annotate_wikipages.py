from SQLite3_utility import create_connection, if_entity_exist_search
from os import listdir
from os.path import join
from random import sample, randrange, choice
import re
from string import ascii_lowercase
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

RE_PATTERN = r'\< ref(.*?)\< \/ref \>'


def upgrade_string(sstring):
    ssplit = sstring.split()
    sstring2 = [word + "-WJM" for word in ssplit]
    return " ".join(sstring2)


def random_flipping(word, if_random):
    if if_random is False:
        return word

    elif not word.isalpha() or len(word) <= 2:
        return word

    else:
        new_word = ""
        for letter in word:
            toss = randrange(100)
            if toss == 0:
                continue
            elif toss == 1:
                new_word += choice(ascii_lowercase)
            else:
                new_word += letter

        return new_word


def sentences_annotation(paragraph, savedir, file_name, if_random):
    if file_name.find("Geography") != -1:
        print("Drop!!!")

    else:
        if paragraph.find("< /ref >") != -1:
            # print(paragraph)
            # print("Find the ref token in :")
            # print(file_name)
            paragraph = re.sub(RE_PATTERN, "", paragraph)
            # print(paragraph)
            # print("\n")
        sens_txt = sent_tokenize(paragraph)
        wf = open(join(savedir, file_name), "w+")

        for sentence in sens_txt:
            # print(len(sentence))
            if sentence.find("-WJM") == -1:
                continue
            tokens = word_tokenize(sentence)
            start_mark = True
            for token in tokens:
                if token.endswith("-WJM") and start_mark is True:
                    wf.write(token.replace("-WJM", "")+"\t"+"B-location"+"\n")
                    start_mark = False
                elif token.endswith(("-WJM")):
                    wf.write(token.replace("-WJM", "") + "\t" + "I-location"+"\n")
                else:
                    token = random_flipping(token, if_random)
                    start_mark = True
                    wf.write(token + "\t" + "O"+"\n")
            wf.write("\n")
        wf.close()


def annotate_link_on_text(db_conn, readdir, savedir1, savedir2, filename="Alaska.txt", location_name="Alaska"):
    rf = open(join(readdir, filename), "r")
    text = rf.readline().rstrip()
    text2 = text

    re_pattern = r'\b({0})\b'.format(location_name)
    if re.search(re_pattern, text):
        text2 = text2.replace(location_name, upgrade_string(location_name))

    link = rf.readline()
    link = rf.readline()
    while link:
        link_text, link_entity = link.rstrip().split("||")
        if if_entity_exist_search(db_conn, link_entity):
            # print(link_text)
            re_pattern = r'\b({0})\b'.format(link_text)
            if re.search(re_pattern, text):
                text2 = text2.replace(link_text, upgrade_string(link_text))
            else:
                break

        link = rf.readline()

    link_text = "United States"
    re_pattern = r'\b({0})\b'.format(link_text)
    if re.search(re_pattern, text):
        text2 = text2.replace(link_text, upgrade_string(link_text))

    sentences_annotation(text2, savedir1, filename, if_random=False)
    sentences_annotation(text2, savedir2, filename, if_random=True)


if __name__ == "__main__":
    dump_page_path = "/home/jiminwan/NeuroTPR_project/WikiTPR/training_data"
    annotate_data_path = "/home/jiminwan/NeuroTPR_project/WikiTPR/training_data2"
    annotate_data_path2 = "/home/jiminwan/NeuroTPR_project/WikiTPR/training_data3"

    db_path = "/home/jiminwan/NeuroTPR_project/WikiTPR/DB/wikiDB.db"
    raw_dataset_path = "/home/jiminwan/NeuroTPR_project/WikiTPR/WikiTPR3000.txt"
    raw_dataset_path2 = "/home/jiminwan/NeuroTPR_project/WikiTPR/WikiTPR3000_randomflipping.txt"
    db_conn = create_connection(db_path)
    print("Successfully connect!")

    filenames = sample(listdir(dump_page_path), 2000)
    for fname in filenames:
        loc_name = (fname.replace('.txt', '')).split(",")[0]
        annotate_link_on_text(db_conn, dump_page_path, annotate_data_path, annotate_data_path2, fname, loc_name)


    with open(raw_dataset_path, "w") as wf:
        for filename in listdir(annotate_data_path):
            if filename.endswith(".txt"):
                text = open(join(annotate_data_path, filename), "r").read()
                wf.write(text)

        wf.close()

    with open(raw_dataset_path2, "w") as wf2:
        for filename in listdir(annotate_data_path2):
            if filename.endswith(".txt"):
                text = open(join(annotate_data_path2, filename), "r").read()
                wf2.write(text)

        wf2.close()
