import gensim
import numpy as np
import pandas as pd
from utils import task1_to_df, task2_2_to_df, task2_1_to_df

path_to_model = './models/GoogleNews-vectors-negative300.bin'

def get_word2vec_embeddings(words):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)

    w2v_embeddings = np.array([])

    for word_original in words:
        word = word_original.replace('-', '_')
        if word in w2v_model.index_to_key:
            embedding = w2v_model.get_vector(word).reshape(1, 300)
        else:
            flag = False
            for w in word.split('_'):
                if w in w2v_model.index_to_key:
                    embedding = w2v_model.get_vector(w).reshape(1, 300)
                    flag = True
                    break
                elif w.replace('.', '', 1).isnumeric() or w.replace(',', '').isnumeric() or w.replace('/', '', 1).isnumeric():
                    embedding = w2v_model.get_vector('numeric').reshape(1, 300)
                    flag = True
                    break
            if not flag:
                embedding = np.zeros(300).reshape(1, 300)
        
        if w2v_embeddings.size == 0:
            w2v_embeddings = embedding
        else:
            w2v_embeddings = np.vstack((w2v_embeddings, embedding))

    return w2v_embeddings

def generate_word_embeddings(df, column, path_to_output_dir):
    words = list(set(df[column]))
    words = [word.replace(' ', '_') for word in words]

    w2v_embeddings = get_word2vec_embeddings(words) 

    for i in range(len(words)):
        np.save(path_to_output_dir + words[i].replace('/', 'div') + '.npy', w2v_embeddings[i]) 

path_to_sentences_dir = './processed_data/parsed_sentences/'
path_to_conll_dir = './processed_data/parsed_conll/'
path_to_file = './data/test/task-{}.txt'
path_to_output_dir_verb = './embeddings/word2vec/verb/'
path_to_output_dir_word = './embeddings/word2vec/word/'

df = task1_to_df(path_to_file.format('1'), path_to_sentences_dir)
generate_word_embeddings(df, 'verb', path_to_output_dir_verb)

df_task2_1 = task2_1_to_df(path_to_file.format('2.1'), path_to_sentences_dir, path_to_conll_dir)
df_task2_2 = task2_2_to_df(path_to_file.format('2.2'), path_to_sentences_dir, path_to_conll_dir)
df = pd.concat([df_task2_1, df_task2_2], ignore_index=True)
generate_word_embeddings(df, 'word', path_to_output_dir_word)