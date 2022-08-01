from os import listdir
from os.path import isfile, join, exists
from collections import defaultdict
import pandas as pd
import conllu

def get_files_from_dir(dir):
    file_names = [f for f in listdir(dir) if isfile(join(dir, f))]
    return file_names

def read_sentences(path_to_sentences_dir):
    sentences_dict = {}
    sentences_files = get_files_from_dir(path_to_sentences_dir)

    for sentence_file in sentences_files:
        file = open(path_to_sentences_dir + sentence_file, 'r')
        sentence = file.readlines()[0].rstrip()
        file.close()

        sentences_dict[sentence_file.split('.')[0]] = sentence
    
    return sentences_dict

def task1_to_df(path_to_file, path_to_sentences_dir):
    if exists('./processed_data/task_1.pkl'):
        return pd.read_pickle('./processed_data/task_1.pkl')

    file = open(path_to_file, 'r')
    samples = [line.rstrip() for line in file.readlines()]
    file.close()

    labels = ['document_id', 'sentence', 'verb_index', 'verb', 'frame', 'frame_id']
    data = []
    
    for sample in samples:
        sample = sample.split(' ')
        document_id = sample[0]
        verb_index = [int(number) for number in sample[1].split('_')]
        verb = " ".join([word for word in sample[2].split('.')[0].split('_')])
        frame = sample[2].split('.')[1]
        frame_id = -1
        file = open(path_to_sentences_dir + document_id + '.txt', 'r')
        sentence = file.readlines()[0].rstrip()
        file.close()
        data.append([document_id, sentence, verb_index, verb, frame, frame_id])

    df = pd.DataFrame(data, columns = labels)
    df.to_pickle('./processed_data/task_1.pkl')
    df.to_excel('./processed_data/task_1.xlsx')

    return df

def task1_df_to_file(df, path_to_file):
    file = open(path_to_file, 'w')
    for index, row in df.iterrows():
        line = row['document_id'] + ' ' + '_'.join([str(verb_index) for verb_index in row['verb_index']]) + ' ' + row['verb'].replace(' ', '_') + '.'
        line += str(row['frame_id'])
        line += '\n'
        file.write(line)
    file.close()

def task2_2_to_df(path_to_file, path_to_sentences_dir, path_to_conll_dir):
    if exists('./processed_data/task_2_2.pkl'):
        return pd.read_pickle('./processed_data/task_2_2.pkl')

    file = open(path_to_file, 'r')
    samples = [line.rstrip() for line in file.readlines()]
    file.close()

    labels = ['document_id', 'instance', 'sentence', 'verb_index', 'verb', 'word_index', 'word', 'xpos', 'deprel', 'before', 'role_id', 'role']
    data = []
    
    for sample in samples:
        sample = sample.split(' ')
        document_id = sample[0]
        verb_index = [int(number) for number in sample[1].split('_')]
        verb = " ".join([word for word in sample[2].split('.')[0].split('_')])
        file = open(path_to_sentences_dir + document_id + '.txt', 'r')
        sentence = file.readlines()[0].rstrip()
        file.close()
        instance = 0

        for i in range(3, len(sample)):
            word = " ".join([word for word in sample[i].split('-:-')[0].split('_')])
            word_index = [int(number) for number in sample[i].split('-:-')[1].split('_')]

            before = 1            
            if word_index[0] > verb_index[0]:
                before = 0

            role = sample[i].split('-:-')[2]
            role_id = -1

            file = open(path_to_conll_dir + document_id + '.txt', "r")
            text = "".join(file.readlines())
            file.close()
            conll = conllu.parse(text)[0]

            xpos = ''
            deprel = ''
            for index in word_index:
                if xpos == '':
                    xpos = conll[index-1]['xpos']
                else:
                    xpos += '_' + conll[index-1]['xpos']
                if deprel == '':
                    deprel = conll[index-1]['deprel']
                else:
                    deprel += '_' + conll[index-1]['deprel']

            instance += 1
            data.append([document_id, instance, sentence, verb_index, verb, word_index, word, xpos, deprel, before, role_id, role])

    df = pd.DataFrame(data, columns = labels)
    df.to_pickle('./processed_data/task_2_2.pkl')
    df.to_excel('./processed_data/task_2_2.xlsx')

    return df

def task2_2_df_to_file(df, path_to_file):
    d = defaultdict(lambda: "")

    for index, row in df.iterrows():
        key = row['document_id'] + '_' + '_'.join([str(verb_index) for verb_index in row['verb_index']]) + '_' + row['verb']
        
        if d[key] == "":
            d[key] = row['document_id'] + ' ' 
            d[key] += '_'.join([str(verb_index) for verb_index in row['verb_index']]) + ' ' 
            d[key] += row['verb'].replace(' ', '_') + '.na' + ' '
            d[key] += row['word'].replace(' ', '_') + '-:-' 
            d[key] += '_'.join([str(word_index) for word_index in row['word_index']]) + '-:-' 
            d[key] += str(row['role_id']) 
        else:
            d[key] += ' '
            d[key] += row['word'].replace(' ', '_') + '-:-' 
            d[key] += '_'.join([str(word_index) for word_index in row['word_index']]) + '-:-' 
            d[key] += str(row['role_id']) 
    
    file = open(path_to_file, 'w')
    for key in d.keys():
        file.write(d[key])
        file.write('\n')
    file.close()

def task2_1_to_df(path_to_file, path_to_sentences_dir, path_to_conll_dir):
    if exists('./processed_data/task_2_1.pkl'):
        return pd.read_pickle('./processed_data/task_2_1.pkl')

    file = open(path_to_file, 'r')
    samples = [line.rstrip() for line in file.readlines()]
    file.close()

    labels = ['document_id', 'instance', 'sentence', 'verb_index', 'verb', 'word_index', 'word', 'xpos', 'deprel', 'before', 'role_id', 'role', 'frame_id', 'frame']
    data = []
    
    for sample in samples:
        sample = sample.split(' ')
        document_id = sample[0]
        verb_index = [int(number) for number in sample[1].split('_')]
        verb = " ".join([word for word in sample[2].split('.')[0].split('_')])
        frame = sample[2].split('.')[1]
        frame_id = -1
        file = open(path_to_sentences_dir + document_id + '.txt', 'r')
        sentence = file.readlines()[0].rstrip()
        file.close()
        instance = 0

        for i in range(3, len(sample)):
            word = " ".join([word for word in sample[i].split('-:-')[0].split('_')])
            word_index = [int(number) for number in sample[i].split('-:-')[1].split('_')]

            before = 1            
            if word_index[0] > verb_index[0]:
                before = 0

            role = sample[i].split('-:-')[2]
            role_id = -1

            file = open(path_to_conll_dir + document_id + '.txt', "r")
            text = "".join(file.readlines())
            file.close()
            conll = conllu.parse(text)[0]

            xpos = ''
            deprel = ''
            for index in word_index:
                if xpos == '':
                    xpos = conll[index-1]['xpos']
                else:
                    xpos += '_' + conll[index-1]['xpos']
                if deprel == '':
                    deprel = conll[index-1]['deprel']
                else:
                    deprel += '_' + conll[index-1]['deprel']

            instance += 1
            data.append([document_id, instance, sentence, verb_index, verb, word_index, word, xpos, deprel, before, role_id, role, frame_id, frame])

    df = pd.DataFrame(data, columns = labels)
    df.to_pickle('./processed_data/task_2_1.pkl')
    df.to_excel('./processed_data/task_2_1.xlsx')

    return df

def task2_1_df_to_file(df, path_to_file):
    d = defaultdict(lambda: "")

    for index, row in df.iterrows():
        key = row['document_id'] + '_' + '_'.join([str(verb_index) for verb_index in row['verb_index']]) + '_' + row['verb']
        
        if d[key] == "":
            d[key] = row['document_id'] + ' ' 
            d[key] += '_'.join([str(verb_index) for verb_index in row['verb_index']]) + ' ' 
            d[key] += row['verb'].replace(' ', '_') + '.' + str(row['frame_id']) + ' '
            d[key] += row['word'].replace(' ', '_') + '-:-' 
            d[key] += '_'.join([str(word_index) for word_index in row['word_index']]) + '-:-' 
            d[key] += str(row['role_id']) 
        else:
            d[key] += ' '
            d[key] += row['word'].replace(' ', '_') + '-:-' 
            d[key] += '_'.join([str(word_index) for word_index in row['word_index']]) + '-:-' 
            d[key] += str(row['role_id']) 
    
    file = open(path_to_file, 'w')
    for key in d.keys():
        file.write(d[key])
        file.write('\n')
    file.close()

def df_copy_frame_ids(df_task1, df_task2_1):
    d = defaultdict(lambda: -1)

    for index, row in df_task1.iterrows():
        key = row['document_id'] + '_' + row['verb'].replace(' ', '_') + '_' + '_'.join([str(index) for index in row['verb_index']])

        if d[key] == -1:
            d[key] = row['frame_id']
        else:
            print(key, d[key], row['frame_id'])

    frame_ids = []
    for index, row in df_task2_1.iterrows():
        key = row['document_id'] + '_' + row['verb'].replace(' ', '_') + '_' + '_'.join([str(index) for index in row['verb_index']])
        frame_ids.append(d[key])
    df_task2_1['frame_id'] = frame_ids

    return df_task2_1
