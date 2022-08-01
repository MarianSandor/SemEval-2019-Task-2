import os
import sklearn
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from vectorize import get_vectors_task_1, get_vectors_task_2_2, get_vectors_task_2_1
from utils import task1_df_to_file, task2_2_df_to_file, task2_1_df_to_file

def generate_submission_task1(df, embedding_name, clusters, affinity, linkage, normalize, layer):
    clusterizer = AgglomerativeClustering(n_clusters=clusters, affinity=affinity, linkage=linkage)

    vecs = get_vectors_task_1(df, embedding_name, layer)
    if normalize:
        vecs = sklearn.preprocessing.normalize(vecs, norm='l2', axis=1)

    df['frame_id'] = clusterizer.fit_predict(vecs)

    if 'layer' in embedding_name:
        embedding_name = embedding_name + '_' + str(layer)

    task1_df_to_file(df, './submissions/task_1/{}_{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage, normalize))

    print('--------------------------------------------Evaluation---------------------------------------------')
    os.system('java -cp ./scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task1 ./data/gold/task-1.txt ./submissions/task_1/{}_{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage, normalize))


def generate_submission_task2_2(df, embedding_name, clusters, affinity, linkage, layer):
    clusterizer = AgglomerativeClustering(n_clusters=clusters, affinity=affinity, linkage=linkage)

    vecs = get_vectors_task_2_2(df, embedding_name, layer)

    df['role_id'] = clusterizer.fit_predict(vecs)

    if 'layer' in embedding_name:
        embedding_name = embedding_name + '_' + str(layer)

    task2_2_df_to_file(df, './submissions/task_2_2/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))

    print('--------------------------------------------Evaluation---------------------------------------------')
    os.system('java -cp ./scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task22 ./data/gold/task-2.2.txt ./submissions/task_2_2/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))
    

def generate_submission_task2_1(df, embedding_name, clusters, affinity, linkage, layer, no_frames):
    clusterizer = AgglomerativeClustering(n_clusters=clusters, affinity=affinity, linkage=linkage)

    vecs = get_vectors_task_2_1(df, embedding_name, layer, no_frames)

    df['role_id'] = clusterizer.fit_predict(vecs)

    if 'layer' in embedding_name:
        embedding_name = embedding_name + '_' + str(layer)

    task2_1_df_to_file(df, './submissions/task_2_1/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))

    print('--------------------------------------------Evaluation---------------------------------------------')
    os.system('java -cp ./scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task21 ./data/gold/task-2.1.txt ./submissions/task_2_1/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))


def generate_submission_task2_1_per_frame(df, embedding_name, clusters, affinity, linkage, layer, no_frames):
    vecs = get_vectors_task_2_1(df, embedding_name, layer, no_frames)

    d = defaultdict(lambda: [])
    for index, row in df.iterrows():
        d[row['frame_id']].append([index, vecs[index]])

    for frame_id in range(no_frames):
        if len(d[frame_id]) == 1:
            df.at[index, 'role_id'] = 0
            continue

        if len(d[frame_id]) > clusters:
            clusterizer = AgglomerativeClustering(n_clusters=clusters, affinity=affinity, linkage=linkage)
        else:
            clusterizer = AgglomerativeClustering(n_clusters=len(d[frame_id]), affinity=affinity, linkage=linkage)

        role_ids = clusterizer.fit_predict(list(map(lambda x: x[1], d[frame_id])))

        for index in range(len(d[frame_id])):
            df.at[index, 'role_id'] = role_ids[index]

    if 'layer' in embedding_name:
        embedding_name = embedding_name + '_' + str(layer)

    task2_1_df_to_file(df, './submissions/task_2_1/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))

    print('--------------------------------------------Evaluation---------------------------------------------')
    os.system('java -cp ./scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task21 ./data/gold/task-2.1.txt ./submissions/task_2_1/{}_{}_{}_{}.txt'.format(embedding_name, clusters, affinity, linkage))