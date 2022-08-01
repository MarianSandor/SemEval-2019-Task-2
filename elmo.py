import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import os
from utils import task1_to_df, task2_2_to_df, task2_1_to_df
import numpy as np
import pandas as pd

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()

path_to_encoder = './models/elmo_3' # https://tfhub.dev/google/elmo/3

def get_elmo_default_embeddings(sentences):
    elmo_model = hub.Module(path_to_encoder, trainable=True)

    elmo_embedding = np.array([])

    print('Total sentences: {}'.format(len(sentences)))
    
    BATCH_SIZE = 10
    for i in range(0, len(sentences), BATCH_SIZE):
        tensors = elmo_model(sentences[i:i+BATCH_SIZE], signature = "default", as_dict = True)['default']

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            embeddings = session.run(tensors)

        print('{}-{} sentences processed...'.format(i, i+BATCH_SIZE))

        if i == 0:
            elmo_embedding = embeddings
        else:
            elmo_embedding = np.vstack((elmo_embedding, embeddings))

    return elmo_embedding

def generate_context_embeddings(df, path_to_output_dir):
    sentences = df['sentence'].values.tolist()

    elmo_embeddings = get_elmo_default_embeddings(sentences)

    for i in range(len(sentences)):
        np.save(path_to_output_dir + df['document_id'][i] + '.npy', elmo_embeddings[i])

def generate_word_embeddings(df, column, path_to_output_dir):
    words = list(set(df[column].values.tolist()))

    elmo_embeddings = get_elmo_default_embeddings(words)

    for i in range(len(words)):
        np.save(path_to_output_dir + words[i].replace(' ', '_').replace('/', 'div') + '.npy', elmo_embeddings[i]) 

path_to_sentences_dir = './processed_data/parsed_sentences/'
path_to_conll_dir = './processed_data/parsed_conll/'
path_to_output_dir_context = './embeddings/elmo/context/'
path_to_output_dir_verb = './embeddings/elmo/verb/'
path_to_output_dir_word = './embeddings/elmo/word/'
path_to_file = './data/test/task-{}.txt'

df = task1_to_df(path_to_file.format('1'), path_to_sentences_dir)
generate_context_embeddings(df, path_to_output_dir_context)
generate_word_embeddings(df, 'verb', path_to_output_dir_verb)

df_task2_1 = task2_1_to_df(path_to_file.format('2.1'), path_to_sentences_dir, path_to_conll_dir)
df_task2_2 = task2_2_to_df(path_to_file.format('2.2'), path_to_sentences_dir, path_to_conll_dir)
df = pd.concat([df_task2_1, df_task2_2], ignore_index=True)
generate_word_embeddings(df, 'word', path_to_output_dir_word)