import numpy as np
from utils import task2_2_to_df

def generate_xpos_embeddings(df, path_to_output_dir):
    labels = []
    for index, row in df.iterrows():
        for label in row['xpos'].split('_'):
            if label not in labels:
                labels.append(label)

    for i in range(len(labels)):
        embedding = np.zeros(len(labels))
        embedding[i] = 1
        np.save(path_to_output_dir + labels[i] + '.npy', embedding)

def generate_deprel_embeddings(df, path_to_output_dir):
    labels = []
    labels_extended = []
    for index, row in df.iterrows():
        for label in row['deprel'].split('_'):
            if label.split(':')[0] not in labels:
                labels.append(label.split(':')[0])
            if label not in labels_extended:
                labels_extended.append(label)

    for i in range(len(labels)):
        embedding = np.zeros(len(labels))
        embedding[i] = 1
        np.save(path_to_output_dir + 'primary/' + labels[i] + '.npy', embedding)
    
    for i in range(len(labels_extended)):
        embedding = np.zeros(len(labels_extended))
        embedding[i] = 1
        np.save(path_to_output_dir + 'extended/' + labels_extended[i].replace(':', '_') + '.npy', embedding)

path_to_file = './data/test/task-{}.txt'
path_to_sentences_dir = './processed_data/parsed_sentences/'
path_to_conll_dir = './processed_data/parsed_conll/'

df = task2_2_to_df(path_to_file.format('2.2'), path_to_sentences_dir, path_to_conll_dir)

path_to_output_dir = './embeddings/xpos/'
generate_xpos_embeddings(df, path_to_output_dir)

path_to_output_dir = './embeddings/deprel/'
generate_deprel_embeddings(df, path_to_output_dir)