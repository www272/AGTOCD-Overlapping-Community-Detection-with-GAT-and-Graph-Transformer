import statistics
import time

import pandas as pd
import skfuzzy as fuzz
from matplotlib.patches import Wedge
from scipy.sparse import coo_matrix
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import nocd
# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import networkx as nx
from matplotlib.colors import to_rgb
import community as community_louvain
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sklearn.preprocessing import minmax_scale
# %matplotlib inline

torch.set_default_tensor_type(torch.cuda.FloatTensor)



def train(beta):
    print(f'Running train() with beta = {beta}')
    f = 'mag_chem'
    loader = nocd.data.load_dataset('data/facebook_ego/fb_348.npz')

    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    N, K = Z_gt.shape
    print(Z_gt)
    print(N)
    print(K)

    hidden_sizes = [128]  # GNN的隐藏层大小
    weight_decay = 1e-2  # GNN权重的L2正则化强度
    dropout = 0.5  # 是否使用dropout
    batch_norm = True  # 是否使用batch normalization`
    lr = 1e-3  # 学习率
    max_epochs = 500  # 训练的最大epoch数
    display_step = 25  # 每多少步计算一次验证损失
    balance_loss = True  # 是否使用平衡损失
    stochastic_loss = True  # 是否使用随机训练（还是全批次训练）
    # stochastic_loss = False  # 是否使用随机训练（还是全批次训练）
    batch_size = 20000  # 批大小（仅对于随机训练有效）

    x_norm = normalize(X)  # 节点特征
    x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()  # 转换为稀疏张量并移至GPU
    sampler = nocd.sampler.get_edge_sampler(A, batch_size, batch_size, num_workers=5)

    # 定义GAT模型
    gnn = nocd.nn.GAT(x_norm.shape[1], hidden_sizes, K, batch_norm=batch_norm, dropout=dropout).cuda()

    # 归一化邻接矩阵
    adj_norm = gnn.normalize_adj(A)
    print(adj_norm.shape)



    decoder = nocd.nn.BerpoDecoder(N, A.nnz, balance_loss=balance_loss)

    # 优化器
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)




    # NMI计算函数
    def get_nmi(thresh=0.5):
        """计算GNN预测的社区的重叠NMI。"""
        gnn.eval()
        Z = F.relu(gnn(x_norm, adj_norm))
        Z_pred = Z.cpu().detach().numpy() > thresh
        nmi = nocd.metrics.overlapping_nmi(Z_pred, Z_gt)
        # cor = compute_cor(Z_pred, Z_gt)
        # print(f'COR = {cor:.4f}')
        return nmi


    def compute_target_distribution(q):
        # q = q.detach().cpu().numpy()

        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()



    max_nmi = -float('inf')
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=10)
    model_saver = nocd.train.ModelSaver(gnn)
    for epoch, batch in enumerate(sampler):
        if epoch >= max_epochs:
            break


        with torch.no_grad():
            gnn.eval()
            # 计算验证损失
            Z = F.relu(gnn(x_norm, adj_norm))
            
            val_loss = decoder.loss_full(Z, A)
            nmi = get_nmi()

            
            if nmi > max_nmi:
                max_nmi = nmi
                best_Z_pred = Z  # 保存最佳社区预测结果
                best_epoch = epoch  # 保存最佳NMI对应的epoch
            print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, nmi = {nmi:.4f},max_nmi = {max_nmi:.4f}')

            # 检查是否需要早停/保存模型
            early_stopping.next_step()
            if early_stopping.should_save():
                model_saver.save()
            if early_stopping.should_stop():
                print(f'Breaking due to early stopping at epoch {epoch}')
                break

        # 训练步骤
        gnn.train()
        opt.zero_grad()
        Z = F.relu(gnn(x_norm, adj_norm))
        ones_idx, zeros_idx = batch
        if stochastic_loss:
            loss = decoder.loss_batch(Z, ones_idx, zeros_idx )
        else:
            loss = decoder.loss_full(Z, A)


        p = compute_target_distribution(Z)
        p_tensor = torch.tensor(p, dtype=torch.float32)

        loss += beta * F.kl_div(Z.log(), p_tensor, reduction='batchmean')
        # loss += beta*kl_cluster_loss(Z)
        loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        opt.step()




    # if best_Z_pred is not None:
    #
    #     plt.figure(figsize=[10, 10])
    #     # 取得最大NMI轮次的社区标签
    #     z = np.argmax(best_Z_pred.cpu().detach().numpy()>0.5, 1)
    #     o = np.argsort(z)
    #     # 使用 nocd 库绘制排序后的稀疏邻接矩阵
    #     nocd.utils.plot_sparse_clustered_adjacency(A, K, z, o, markersize=0.25)
    #     plt.show()


if __name__ == '__main__':
    for beta in [0.001]:  # 注意不能使用 set，会导致顺序不确定
        for epoch in range(10):
            train(beta)