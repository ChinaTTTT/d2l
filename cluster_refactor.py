#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F


class CosineSimilarity(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.dim, self.eps)


# 余弦相似度聚类
def cos_sim_cluster(sim_matrix, data):
    # 余弦相似度聚类
    label_pred = []
    for i in range(sim_matrix.shape[0]):
        label_pred.append(i)
    for i in range(sim_matrix.shape[0]):
        print(f'labeling {i}')
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i][j] > 0.8:
                label_pred[j] = label_pred[i]
    # 增加分类标签这一列
    data['label'] = label_pred
    return data.sort_values(by='label')


def cos_sim(emb1, emb2):
    emb1 = torch.tensor(emb1).to(device)
    emb2 = torch.tensor(emb2).to(device)
    model = CosineSimilarity().to(device)
    model = torch.nn.DataParallel(model)

    start_time = time.time()
    tensor_list = []
    for i in range(emb1.shape[0]):
        print(i)
        x1 = emb1[i]
        x2 = emb2
        distance = model(x1.unsqueeze(0), x2)
        tensor_list.append(distance)
    distance = torch.cat([t.unsqueeze(0) for t in tensor_list], dim=0)
    end_time = time.time()
    print("cos_sim time is :{}".format(end_time - start_time))
    return distance  # Tensor


def cos_sim_label(emb1, emb2, data):
    emb1 = torch.tensor(emb1).to(device)
    emb2 = torch.tensor(emb2).to(device)
    model = CosineSimilarity().to(device)
    model = torch.nn.DataParallel(model)

    start_time = time.time()
    label_pred = []
    for i in range(emb1.shape[0]):
        label_pred.append(i)
    for i in range(emb1.shape[0]):
        print(i)
        x1 = emb1[i]
        x2 = emb2
        distance = model(x1.unsqueeze(0), x2)
        distance = distance.cpu().detach().numpy()
        for j in range(i + 1, len(distance)):
            if distance[j] > 0.7:
                label_pred[j] = label_pred[i]
    # 增加分类标签这一列
    data['label'] = label_pred
    end_time = time.time()
    print("cos_sim time is :{}".format(end_time - start_time))
    return data.sort_values(by='label')  # Tensor


def main(root_path, file, model):
    s_time = time.time()
    data = pd.read_csv(root_path + file)
    ques_data = data['question']
    emb = model.encode(ques_data.tolist())
    print(f'{file} embedding完成！')

    cluster_data = cos_sim_label(emb, emb, data)
    grouped_data = cluster_data.reset_index(drop=True)
    grouped_data.to_excel(f'E:/Customer_demo/xp-algo-know-qa-gen/embedding/clustered2.xlsx', index=False)
    print(f'{file}聚类完成！')
    print(f'耗时{time.time() - s_time}s!')
    return


if __name__ == '__main__':
    import os
    root_path = 'E:/Customer_demo/xp-algo-know-qa-gen/embedding/'
    file_list = os.listdir(root_path)
    file_list = [file for file in file_list if 'QA' in file]
    model = SentenceTransformer('C:/Users/xpeng/Downloads/2023110701/best')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for file in file_list[:]:
        main(root_path, file, model)
    print('All Done!')
    
    