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
    emb1 = torch.tensor(emb1)
    emb2 = torch.tensor(emb2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CosineSimilarity().to(device)
    model = torch.nn.DataParallel(model)

    start_time = time.time()
    tensor_list = []
    for i in range(emb1.shape[0]):
        print(i)
        x1 = emb1[i].to(device)
        x2 = emb2.to(device)
        distance = model(x1, x2)
        tensor_list.append(distance)
    distance = torch.cat([t.unsqueeze(0) for t in tensor_list], dim=0)
    end_time = time.time()
    print("cos_sim time is :{}".format(end_time - start_time))
    return distance  # Tensor


def cos_sim_label(emb1, emb2, data):
    emb1 = torch.tensor(emb1)
    emb2 = torch.tensor(emb2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CosineSimilarity().to(device)
    model = torch.nn.DataParallel(model)

    start_time = time.time()
    label_pred = []
    for i in range(emb1.shape[0]):
        label_pred.append(i)
    for i in range(emb1.shape[0]):
        print(i)
        x1 = emb1[i].to(device)
        x2 = emb2.to(device)
        distance = model(x1, x2)
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
    #month = file.split('_')[-1].split('.')[0]
    data = pd.read_csv(root_path + file)
    # data = data[(data['first_pay_time'].isnull()) & (data['own_veh_cnt'].isnull())]
    # data = data.drop_duplicates(subset='question')
    # embedding
    ques_data = data['question']
    emb = model.encode(ques_data.tolist())
    print(f'{file} embedding完成！')

    # sim_matrix = cos_sim(emb, emb)

    # print(f'{file}相似度计算完成！')
    # sim_matrix = sim_matrix.cpu().detach().numpy()
    cluster_data = cos_sim_label(emb, emb, data)
    # grouped_data = cluster_data.groupby('label').agg({'session_id': list, 'uid': list, 'question': 'last', 'answer': 'last'})
    # grouped_data = cluster_data.sort_values('ds', ascending=False).groupby('label').head(1)
    grouped_data = cluster_data.reset_index(drop=True)
    # grouped_data = grouped_data.drop(columns=['label'])
    grouped_data.to_excel(f'E:/Customer_demo/xp-algo-know-qa-gen/embedding/clustered.xlsx', index=False)
    print(f'{file}聚类完成！')
    print(f'耗时{time.time() - s_time}s!')
    return


if __name__ == '__main__':
    # 读取数据
    import os

    # 获取目录下所有文件
    root_path = 'E:/Customer_demo/xp-algo-know-qa-gen/embedding/'
    file_list = os.listdir(root_path)
    file_list = [file for file in file_list if 'QA' in file]#
    model = SentenceTransformer('C:/Users/xpeng/Downloads/2023110701/best')
    for file in file_list[:]:
        main(root_path, file, model)
    print('All Done!')


##