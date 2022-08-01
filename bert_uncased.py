from regex import E
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from utils import task1_to_df, task2_2_to_df, task2_1_to_df

path_to_preprocess = './models/bert_en_uncased_preprocess_3' # https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
path_to_encoder = './models/bert_en_uncased_L-12_H-768_A-12_4' # https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4

def get_bert_embeddings(sentences, output, layer = 11):
    bert_preprocess_model = hub.KerasLayer(path_to_preprocess)
    bert_model = hub.KerasLayer(path_to_encoder)

    bert_embedding = np.array([])

    print('Total sentences: {}'.format(len(sentences)))
    
    BATCH_SIZE = 50
    for i in range(0, len(sentences), BATCH_SIZE):
        sentences_preprocessed = bert_preprocess_model(sentences[i:i+BATCH_SIZE])
        bert_partial_result = bert_model(sentences_preprocessed)

        print('{}-{} sentences processed...'.format(i, i+BATCH_SIZE))

        if i == 0:
            if output == 'encoder_outputs':
                bert_embedding = bert_partial_result[output][layer].numpy()
            else:
                bert_embedding = bert_partial_result[output].numpy()
        else:
            if output == 'encoder_outputs':
                bert_embedding = np.vstack((bert_embedding, bert_partial_result[output][layer].numpy()))
            else:
                bert_embedding = np.vstack((bert_embedding, bert_partial_result[output].numpy()))

    return bert_embedding

def generate_context_embeddings_pooled_output(df, path_to_output_dir):
    sentences = df['sentence'].values.tolist()

    if not os.path.isdir(path_to_output_dir + 'pooled_output'):
        os.mkdir(path_to_output_dir + 'pooled_output/')
    else:
        return

    bert_embedding = get_bert_embeddings(sentences, 'pooled_output')

    for i in range(len(sentences)):
        np.save(path_to_output_dir + 'pooled_output/' + df['document_id'][i] + '.npy', bert_embedding[i])

def generate_context_embeddings_by_layer(df, layer, path_to_output_dir):
    sentences = df['sentence'].values.tolist()

    if not os.path.isdir(path_to_output_dir + 'layer_' + str(layer)):
        os.mkdir(path_to_output_dir + 'layer_' + str(layer))
    else:
        return

    bert_embedding = get_bert_embeddings(sentences, 'encoder_outputs', layer)

    for i in range(len(sentences)):
        layer_encodding = np.mean(bert_embedding[i], axis = 0)
        np.save(path_to_output_dir + 'layer_' + str(layer) + '/' + df['document_id'][i] + '.npy', layer_encodding)

def generate_word_embeddings(df, column, path_to_output_dir):
    words = list(set(df[column].values.tolist()))

    bert_embedding = get_bert_embeddings(words, 'pooled_output')

    for i in range(len(words)):
        np.save(path_to_output_dir + words[i].replace(' ', '_').replace('/', 'div') + '.npy', bert_embedding[i])    

path_to_sentences_dir = './processed_data/parsed_sentences/'
path_to_conll_dir = './processed_data/parsed_conll/'
path_to_output_dir_context = './embeddings/bert_uncased/context/'
path_to_output_dir_verb = './embeddings/bert_uncased/verb/'
path_to_output_dir_word = './embeddings/bert_uncased/word/'
path_to_file = './data/test/task-{}.txt'

df = task1_to_df(path_to_file.format('1'), path_to_sentences_dir)
generate_context_embeddings_pooled_output(df, path_to_output_dir_context)
generate_context_embeddings_by_layer(df, 3, path_to_output_dir_context)
generate_context_embeddings_by_layer(df, 6, path_to_output_dir_context)
generate_context_embeddings_by_layer(df, 9, path_to_output_dir_context)
generate_word_embeddings(df, 'verb', path_to_output_dir_verb)

df_task2_1 = task2_1_to_df(path_to_file.format('2.1'), path_to_sentences_dir, path_to_conll_dir)
df_task2_2 = task2_2_to_df(path_to_file.format('2.2'), path_to_sentences_dir, path_to_conll_dir)
df = pd.concat([df_task2_1, df_task2_2], ignore_index=True)
generate_word_embeddings(df, 'word', path_to_output_dir_word)