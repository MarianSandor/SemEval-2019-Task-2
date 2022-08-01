import conllu
from os import listdir
from os.path import isfile, join
from utils import get_files_from_dir

def split_conll_file(file_path, output_dir):
    file = open(file_path, "r")
    samples = file.readlines()
    samples = "".join(samples)
    samples = samples.split("\n\n")

    for sample in samples:
        sample = sample.split("\n", 1)
        id = sample[0]
        text = sample[1]

        output_file_path = output_dir + id + ".txt"
        output_file = open(output_file_path, "w")
        output_file.write(text)
        output_file.close()

    file.close()

def conll_to_sentence(file_path):
    file = open(file_path, "r")
    text = "".join(file.readlines())

    sentence = ""
    for token in conllu.parse(text)[0]:
        sentence += token["lemma"]
        sentence += " "
    sentence = sentence[:-1]

    file.close()

    return sentence

def parse_conlls(dir, output_dir):
    conll_files = get_files_from_dir(dir)

    for conll_file in conll_files:
        sentence = conll_to_sentence(dir + conll_file)

        file_out = open(output_dir + conll_file, "w")
        file_out.write(sentence)
        file_out.close()

path_to_conll_file = './data/dep-stx/pos-gold-dep-auto.conll.txt'
path_to_parsed_conll_dir = './processed_data/parsed_conll/'
path_to_parsed_sentences_dir = './processed_data/parsed_sentences/'

print('split_conll_file...')
split_conll_file(path_to_conll_file, path_to_parsed_conll_dir);

print('parse_conlls...')
parse_conlls(path_to_parsed_conll_dir, path_to_parsed_sentences_dir)