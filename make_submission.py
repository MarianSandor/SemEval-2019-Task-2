import sys
from utils import task1_to_df, task2_2_to_df, task2_1_to_df, df_copy_frame_ids
from submission import generate_submission_task1, generate_submission_task2_2, generate_submission_task2_1

task1 = [
    #70.85
    ['elmo_context_elmo_word', 155, 'euclidean', 'complete', True, None],             
    #70.28
    ['bert_cased_context_layer_elmo_word', 160, 'manhattan', 'complete', True, 3],  
    #70.24
    ['bert_cased_context_layer_elmo_word', 165, 'manhattan', 'complete', True, 3],    
    #70.00
    ['bert_uncased_context_layer_elmo_word', 155, 'euclidean', 'complete', True, 3]
]

task2_1 = [
    #56.14
    ['features_only', ['_frame', '_before'], 315, 'euclidean', 'complete', None],
    #51.61
    ['features_only', ['_frame', '_before', '_pos'], 310, 'euclidean', 'complete', None],
    #48.97
    ['features_only', ['_frame', '_deprel', '_before'], 2, 'euclidean', 'ward', None],
    #46.59
    ['elmo_word', ['_frame', '_xpos'], 5, 'euclidean', 'ward', None],
    #46.59
    ['bert_cased_context_layer_elmo_word', ['_frame', '_before', '_pos'], 5, 'euclidean', 'ward', 3],
    #46.45
    ['bert_uncased_context_layer_elmo_word', ['_frame', '_xpos', '_before', '_pos'], 5, 'euclidean', 'ward', 3],
    #46.36
    ['elmo_context_elmo_word', ['_frame', '_deprel'], 5, 'euclidean', 'ward', None],
]

task2_2 = [
    #44.16
    ['elmo_context_elmo_word', ['_xpos'], 5, 'euclidean', 'ward', None],                                 
    #44.15
    ['bert_cased_context_layer_elmo_word', ['_xpos', '_before', '_pos'], 5, 'euclidean', 'ward', 3],     
    #43.98
    ['bert_uncased_context_layer_elmo_word', [], 5, 'euclidean', 'ward', 3],                              
    #43.88
    ['elmo_context_elmo_word', ['_deprel'], 5, 'euclidean', 'ward', None],      
    #43.83
    ['bert_uncased_context_pooled_output_elmo_word', ['_deprel', '_before', '_pos'], 5, 'euclidean', 'ward', None],         
    #43.82                         
    ['bert_cased_context_pooled_output_elmo_word', ['_xpos', '_before', '_pos'], 5, 'euclidean', 'ward', None],                    
]

if len(sys.argv) != 4:
    print('Ussage: {} task1_id task2_1_id task2_2_id'.format(sys.argv[0]))

task1_id = int(sys.argv[1])
task2_1_id = int(sys.argv[2])
task2_2_id = int(sys.argv[3])

if task1_id < 0 or task1_id >= len(task1):
    print('Invalid task1_id!')

if task2_1_id < 0 or task2_1_id >= len(task2_1):
    print('Invalid task2_1_id!')

if task2_2_id < 0 or task2_2_id >= len(task2_2):
    print('Invalid task2_2_id!')

path_to_sentences_dir = './processed_data/parsed_sentences/'
path_to_conll_dir = './processed_data/parsed_conll/'
path_to_file = './data/gold/task-{}.txt'

df_task1 = task1_to_df(path_to_file.format('1'), path_to_sentences_dir)
df_task2_2 = task2_2_to_df(path_to_file.format('2.2'), path_to_sentences_dir, path_to_conll_dir)
df_task2_1 = task2_1_to_df(path_to_file.format('2.1'), path_to_sentences_dir, path_to_conll_dir)

print('Task1...')
print(task1[task1_id])
generate_submission_task1(df_task1, task1[task1_id][0], task1[task1_id][1], task1[task1_id][2], task1[task1_id][3], task1[task1_id][4], task1[task1_id][5])

print('Task2_1...')
print(task2_1[task2_1_id])
df_task2_1 = df_copy_frame_ids(df_task1, df_task2_1)
generate_submission_task2_1(df_task2_1, task2_1[task2_1_id][0] + ''.join(task2_1[task2_1_id][1]), task2_1[task2_1_id][2], task2_1[task2_1_id][3], task2_1[task2_1_id][4], task2_1[task2_1_id][5], task1[task1_id][1])

print('Task2_2...')
print(task2_2[task2_2_id])
generate_submission_task2_2(df_task2_2, task2_2[task2_2_id][0] + ''.join(task2_2[task2_2_id][1]), task2_2[task2_2_id][2], task2_2[task2_2_id][3], task2_2[task2_2_id][4], task2_2[task2_2_id][5])