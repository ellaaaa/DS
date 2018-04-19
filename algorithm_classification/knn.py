import pandas as pd

df = pd.read_csv(r'C:\Users\Administrator\Desktop\HM\movies.csv')

def knn_classify(target, kind, data_set, k):
    data_size = len(kind)
    diff = pd.DataFrame([target] * data_size) - data_set
    sq_diff = diff ** 2
    sq_dis = sq_diff.sum(axis=1)
    dis = sq_dis ** 0.5
    sorted_dis = dis.sort_values()

    class_count = {}
    for i in range(k):
        vote_kind = kind[sorted_dis.index[i]]
        class_count[vote_kind] = class_count.get(vote_kind, 0) + 1

    sorted_class_count = sorted(class_count.items())
    return sorted_class_count[0][0]

print(knn_classify([30,10],df['Kind'], df[['Movement','kiss']],1))