# book 'machine learning in action' wriiten bu Pter Harrington
# book 'machine learning' wrriten by Zhou Zhihua
# book 'Introduction to Data Mining' written by Tan, Steinbach, Kumkar

from math import log

def shannon_ent(data_set):
    '''calculate shannon entropy'''
    shannon_ent = 0.0
    num_feat = len(data_set.index)
    if type(data_set) is 'DataFrame':
        labels = data_set[data_set.columns[-1]].value_counts()
    else:
        labels = data_set.value_counts()
    for label_counts in labels:
        prob = float(label_counts)/num_feat
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def best_feat_split(data_set):
    '''choose best feature to splite'''
    global best_feat
    base_ent = shannon_ent(data_set)
    best_gain = 0.0
    for feat in data_set.columns[:-1]:
        unique = data_set[feat].value_counts()
        new_ent = 0.0
        for v in unique:
            sub_data_set = data_set[data_set[feat] == v]
            prob = len(sub_data_set)/float(len(data_set.index))
            new_ent += prob*shannon_ent(sub_data_set)
        info_gain = base_ent - new_ent
        if info_gain > best_gain:
            best_gain = info_gain
            best_feat = feat
    return best_feat

