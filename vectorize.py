import numpy as np
from os.path import exists

path_to_context_embeddings = './embeddings/{}/context/'
path_to_verb_embeddings = './embeddings/{}/verb/'
path_to_word_embeddings = './embeddings/{}/word/'
path_to_feature_embeddings = './embeddings/{}/'

def get_vectors_task_1(df, embedding_name, layer = None):
    if 'layer' in embedding_name:
        if exists('./vectors/task1/' + embedding_name + '_' + str(layer) + '.npy'):
            return np.load('./vectors/task1/' + embedding_name + '_' + str(layer) + '.npy')
    else:
        if exists('./vectors/task1/' + embedding_name + '.npy'):
            return np.load('./vectors/task1/' + embedding_name + '.npy')

    vecs = []

    if embedding_name == 'bert_cased_context_pooled_output':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            vecs.append(context)

    if embedding_name == 'bert_cased_context_layer':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            vecs.append(context)

    if embedding_name == 'bert_uncased_context_pooled_output':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            vecs.append(context)

    if embedding_name == 'bert_uncased_context_layer':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            vecs.append(context)

    if embedding_name == 'elmo_context':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            vecs.append(context)
#==============================================================================================================================================
# Bert Cased Context Pooled Output + Word    
    if embedding_name == 'bert_cased_context_pooled_output_bert_cased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_cased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_cased_context_pooled_output_elmo_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_cased_context_pooled_output_word2vec_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('word2vec') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
# Bert Cased Context Layer + Word      
    if embedding_name == 'bert_cased_context_layer_bert_cased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_cased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_cased_context_layer_elmo_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_cased_context_layer_word2vec_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('word2vec') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
# Bert Uncased Context Pooled Output + Word  
    if embedding_name == 'bert_uncased_context_pooled_output_bert_uncased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_uncased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_uncased_context_pooled_output_elmo_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_uncased_context_pooled_output_word2vec_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('word2vec') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
# Bert Uncased Context Layer + Word    
    if embedding_name == 'bert_uncased_context_layer_bert_uncased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_uncased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_uncased_context_layer_elmo_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'bert_uncased_context_layer_word2vec_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('word2vec') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
# Elmo Contex + Word
    if embedding_name == 'elmo_context_elmo_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'elmo_context_bert_cased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_cased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name == 'elmo_context_bert_uncased_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('bert_uncased') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
    
    if embedding_name == 'elmo_context_word2vec_word':
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_verb_embeddings.format('word2vec') + row['verb'].replace(' ', '_') + '.npy')
            vecs.append(np.hstack((context, word)))
#==============================================================================================================================================
    

    if 'layer' in embedding_name:
        np.save('./vectors/task1/' + embedding_name + '_' + str(layer) + '.npy', vecs)
    else:
        np.save('./vectors/task1/' + embedding_name + '.npy', vecs)

    return vecs


def get_vectors_task_2_2(df, embedding_name, layer = None):
    if 'layer' in embedding_name:
        if exists('./vectors/task2_2/' + embedding_name + '_' + str(layer) + '.npy'):
            return np.load('./vectors/task2_2/' + embedding_name + '_' + str(layer) + '.npy')
    else:
        if exists('./vectors/task2_2/' + embedding_name + '.npy'):
            return np.load('./vectors/task2_2/' + embedding_name + '.npy')

    vecs = []

    if embedding_name.startswith('bert_cased_context_pooled_output_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_cased_context_pooled_output_word2vec_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('word2vec') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_cased_context_layer_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_cased_context_layer_word2vec_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('word2vec') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))
    
    if embedding_name.startswith('bert_cased_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('bert_cased') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)
#----------------------------------------------------------------------------------------------------------------------------------------------
    if embedding_name.startswith('bert_uncased_context_pooled_output_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_uncased_context_pooled_output_word2vec_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'pooled_output/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('word2vec') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_uncased_context_layer_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_uncased_context_layer_word2vec_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('word2vec') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_uncased_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('bert_uncased') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)
#----------------------------------------------------------------------------------------------------------------------------------------------
    if embedding_name.startswith('elmo_context_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))
    
    if embedding_name.startswith('elmo_context_word2vec_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('word2vec') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('elmo_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)
#==============================================================================================================================================
    vecs = np.array(vecs)

    vecs = concat_features(df, embedding_name, vecs)

    if 'layer' in embedding_name:
        np.save('./vectors/task2_2/' + embedding_name + '_' + str(layer) + '.npy', vecs)
    else:
        np.save('./vectors/task2_2/' + embedding_name + '.npy', vecs)

    return vecs


def get_vectors_task_2_1(df, embedding_name, layer = None, no_frames = None):
    if 'layer' in embedding_name:
        if exists('./vectors/task2_1/' + embedding_name + '_' + str(layer) + '.npy'):
            return np.load('./vectors/task2_1/' + embedding_name + '_' + str(layer) + '.npy')
    else:
        if exists('./vectors/task2_1/' + embedding_name + '.npy'):
            return np.load('./vectors/task2_1/' + embedding_name + '.npy')

    vecs = []

    if embedding_name.startswith('bert_cased_context_layer_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('bert_uncased_context_layer_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))

    if embedding_name.startswith('elmo_context_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
    if embedding_name.startswith('bert_cased_context_layer_elmo_verb_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_cased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            verb = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, verb, word)))

    if embedding_name.startswith('bert_uncased_context_layer_elmo_verb_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('bert_uncased') + 'layer_' + str(layer) + '/' + row['document_id'] + '.npy')
            verb = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, verb, word)))

    if embedding_name.startswith('elmo_context_elmo_verb_elmo_word'):
        for index, row in df.iterrows():
            context = np.load(path_to_context_embeddings.format('elmo') + row['document_id'] + '.npy')
            verb = np.load(path_to_verb_embeddings.format('elmo') + row['verb'].replace(' ', '_') + '.npy')
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(np.hstack((context, verb, word)))
#----------------------------------------------------------------------------------------------------------------------------------------------
    if embedding_name.startswith('bert_cased_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('bert_cased') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)

    if embedding_name.startswith('bert_uncased_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('bert_uncased') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)

    if embedding_name.startswith('elmo_word'):
        for index, row in df.iterrows():
            word = np.load(path_to_word_embeddings.format('elmo') + row['word'].replace(' ', '_').replace('/', 'div') + '.npy')
            vecs.append(word)
#==============================================================================================================================================
    vecs = np.array(vecs)

    vecs = concat_features(df, embedding_name, vecs, no_frames)

    if 'layer' in embedding_name:
        np.save('./vectors/task2_1/' + embedding_name + '_' + str(layer) + '.npy', vecs)
    else:
        np.save('./vectors/task2_1/' + embedding_name + '.npy', vecs)

    return vecs


def concat_features(df, embedding_name, vecs, no_frames=None):

    if '_frame' in embedding_name:
        frames = np.array([])
        for index, row in df.iterrows():
            v = np.zeros(no_frames)
            v[row['frame_id']] = 1
            v = v.astype(int)
            if frames.size == 0:
                frames = v
            else:
                frames = np.vstack((frames, v))
        frames = frames.astype(int)
        if vecs.size == 0:
            vecs = frames
        else:
            vecs = np.hstack((vecs, frames))

    if '_xpos' in embedding_name:
        xpos = np.array([])
        for index, row in df.iterrows():
            v = np.array([])
            for label in row['xpos'].split('_'):
                if v.size == 0:
                    v = np.load(path_to_feature_embeddings.format('xpos') + label + '.npy').astype(int)
                else:
                    v = np.add(v, np.load(path_to_feature_embeddings.format('xpos') + label + '.npy').astype(int))
                break
            v = np.vectorize(lambda x: min(1, x))(v)
            if xpos.size == 0:
                xpos = v
            else:
                xpos = np.vstack((xpos, v))
        xpos = xpos.astype(int)
        if vecs.size == 0:
            vecs = xpos
        else:
            vecs = np.hstack((vecs, xpos))


    if '_deprel' in embedding_name:
        deprel = np.array([])
        for index, row in df.iterrows():
            v = np.array([])
            for label in row['deprel'].split('_'):
                if v.size == 0:
                    v = np.load(path_to_feature_embeddings.format('deprel') + 'primary/' + label.split(':')[0] + '.npy').astype(int)
                else:
                    v = np.add(v, np.load(path_to_feature_embeddings.format('deprel') + 'primary/' + label.split(':')[0] + '.npy').astype(int))
                break
            v = np.vectorize(lambda x: min(1, x))(v)
            if deprel.size == 0:
                deprel = v
            else:
                deprel = np.vstack((deprel, v))
        deprel = deprel.astype(int)
        if vecs.size == 0:
            vecs = deprel
        else:
            vecs = np.hstack((vecs, deprel))

    if '_before' in embedding_name:
        v = np.array(list(df['before'])).reshape(len(df), 1)
        v = v.astype(int)
        if vecs.size == 0:
            vecs = v
        else:
            vecs = np.hstack((vecs, v))

    if '_pos' in embedding_name:
        v = np.array(list(df['instance'])).reshape(len(df), 1)
        v = v.astype(int)
        if vecs.size == 0:
            vecs = v
        else:
            vecs = np.hstack((vecs, v))

    return vecs