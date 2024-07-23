from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import random
import ipdb


class CA_PCAer(object):
    def __init__(self, dim=64):
        '''
        input: dim, dimension after PCA
        '''
        self.pca = PCA(n_components=dim)


    def train(self, db):
        '''
        训练PCA模型
        input: data: input array, shape = [N, D]
            N: sample number
            D: feature dimension

        output: PCA data, database data after PCA
        '''
        y_db = self.pca.fit_transform(db)
        return y_db


    def forward(self, query):
        '''
        压缩一个新特征向量
        input: data: input array, shape = [N, D] or [D]
            N: sample number
            D: feature dimension

        output: PCA data, query data after PCA
        '''
        if len(query.shape) == 1:
            y_query = self.pca.transform(query[np.newaxis, :]).reshape(-1)
        else:
            y_query = self.pca.transform(query)
        return y_query


def calculate_r_at_1(pca_db, pca_query):
    '''
    计算 R@1 accuracy
    input: pca_db: PCA-transformed database data
           pca_query: PCA-transformed query data
    output: R@1 accuracy
    '''
    similarities = cosine_similarity(pca_query, pca_db)
    nearest_neighbors = np.argmax(similarities, axis=1)
    correct_matches = (nearest_neighbors == random_indices)
    render_list(nearest_neighbors, random_indices)
    return np.mean(correct_matches)


def render_list(list1, list2, n=None):
    assert len(list1) == len(list2)
    n = n or len(list1)
    for elem1, elem2 in zip(list1, list2):
        if n == 0:
            break
        print(f"{elem1}\t{elem2}" if elem1 == elem2 else f"{elem1}\t{elem2}\t*")
        n -= 1


if __name__ == '__main__':
    data = 2.5 * (2 * np.random.rand(100, 256) - 1)     # 随机data生成数据
    random_indices = [random.randint(0, len(data) - 1) for _ in range(50)]
    query = data[random_indices]                        # 随机query生成数据

    ca = CA_PCAer(32)                                   # 初始化PCA器, 降维到32维

    pca_data = ca.train(data)                           # 用data训练PCA器
    pca_query = ca.forward(query)                       # 用训练好的PCA器对query做PCA处理

    r_at_1_accuracy = calculate_r_at_1(pca_data, pca_query) # 计算 R@1 accuracy
    print(f'R@1 Accuracy: {r_at_1_accuracy:.4f}')
    