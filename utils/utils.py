# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import datetime
import importlib
import random

import numpy as np
import pandas as pd
import torch


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(alias=None, is_federated=False):
    if is_federated:
        model_name = alias.lower()
        module_path = '.'.join(['models', model_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)

        return getattr(model_module, '{}Trainer'.format(alias))
    else:
        return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.4f' % value + ', '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def sampleClients(client_list, sample_strategy='random', sample_ratio=0.1, last_clients=None):
    """
    Sample clients from the client list.
    :param client_list: a list of client ids
    :param sample_strategy: str, the sample strategy
    :param sample_ratio: float, the sample ratio
    :param last_clients: a list of client ids
    :return: a list of client ids
    """

    if sample_ratio == 1:
        # Use all clients
        participants = client_list
    else:
        # Calculate the number of clients to be sampled
        sample_num = int(len(client_list) * sample_ratio)

        # Remove the clients that have been sampled in the last round
        if last_clients is not None:
            client_list = list(set(client_list) - set(last_clients))

        if sample_strategy == 'random':
            # Randomly sample clients
            participants = random.sample(client_list, sample_num)
        else:
            raise ValueError('Invalid sample strategy: {}'.format(sample_strategy))

    return participants


def get_combinations(config, result_file):
    """
    Get all combinations of hyperparameters.
    :param config: dict, contains all configurations
    :param result_file: str, the Excel file where results are saved
    :return: a list of dict, the combinations of hyperparameters
    """
    from itertools import product

    # Step 1: 尝试读取现有的 Excel 文件，获取已经运行过的参数组合
    try:
        df = pd.read_csv(result_file)
        existing_combinations = df[config['hyper_parameters']].to_dict('records')  # 已存在的组合
    except FileNotFoundError:
        existing_combinations = []

    # hyper-parameters
    hyper_ls = []
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))

    # Step 3: 排除已经运行过的参数组合
    # 将所有组合转为字典格式以便比较
    combination_dicts = [dict(zip(config['hyper_parameters'], comb)) for comb in combinators]

    # 过滤掉已经存在的组合
    remaining_combinations = [
        comb for comb in combination_dicts if comb not in existing_combinations
    ]

    comb_tuple = [list(d.values()) for d in remaining_combinations]

    total_loops = len(remaining_combinations)

    return comb_tuple, total_loops


def save_experiment_results(param_dict, result_dict, csv_filename='experiment_results.csv'):
    # 合并参数和实验结果为一条记录
    record = {**param_dict, **result_dict}

    # 尝试读取现有的Excel文件，如果文件不存在，则创建一个新的DataFrame
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 将参数字典转为DataFrame行
    new_row = pd.DataFrame([record])

    # 检查是否存在相同的参数组合
    if not df.empty and all(col in df.columns for col in param_dict):
        # 找到具有相同参数组合的行
        match = df.loc[(df[list(param_dict)] == pd.Series(param_dict)).all(axis=1)]

        if not match.empty:
            # 如果找到了相同的参数组合，获取行索引
            index = match.index[0]
            df.loc[index] = new_row.loc[0]
        else:
            # 如果没有找到相同的参数组合，添加新行
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # 如果文件是新的或没有匹配列，直接追加数据
        df = pd.concat([df, new_row], ignore_index=True)

    # 将结果写入Excel文件
    df.to_csv(csv_filename, index=False)


def find_best_parameters(csv_filename='experiment_results.csv', metric='ndcg@10', maximize=True):
    """
    从CSV文件中读取所有实验结果，并根据指定的评价指标选取最优的参数组合。
    :param csv_filename: str, 存储实验结果的CSV文件名
    :param metric: str, 需要优化的评价指标名称
    :param maximize: bool, 如果为True，表示需要最大化指标；如果为False，表示需要最小化指标
    :return: dict, 包含最优参数组合及其对应的评价指标
    """
    # 从CSV文件读取数据
    df = pd.read_csv(csv_filename)

    # 检查指定的评价指标是否存在
    if metric not in df.columns:
        raise ValueError(f"指定的评价指标 '{metric}' 不存在，请检查 CSV 文件中的列名。")

    # 根据评价指标选择最优的参数组合
    if maximize:
        best_row = df.loc[df[metric].idxmax()]  # 最大化指标
    else:
        best_row = df.loc[df[metric].idxmin()]  # 最小化指标

    # 提取最优参数组合及其评价指标
    best_parameters = best_row.to_dict()

    return best_parameters


def mail_notice(config, content=None):
    """
    Email notice that the training is finished.
    :param config: the input arguments
    :return: None
    """
    import iMail
    from configs import private as uc
    import yaml

    # Set the Mail Sender
    mail_obj = iMail.EMAIL(host=uc.EMAIL_HOST, sender_addr=uc.SENDER_ADDRESS, pwd=uc.PASSWORD,
                           sender_name=uc.SENDER_NAME)
    mail_obj.set_receiver(uc.RECEIVER)

    # Create a new email
    mail_title = '[NOTICE FROM EXPERIMENT] {} on {}: {}-{}'.format(
        config['model'], config['dataset'].upper(), config['type'], config['comment']
    )
    mail_obj.new_mail(subject=mail_title, encoding='UTF-8')

    # Attach a text to the receiver
    mail_obj.add_text(content=yaml.dump(content))

    # Send the email
    mail_obj.send_mail()

    return mail_title, content


def dp_step(param, threshold=1.0, sigma=1.0):

    grad_norm = torch.norm(param).item()
    clip_value = threshold / max(1.0, grad_norm / threshold)
    param.data.mul_(clip_value)  # 梯度裁剪

    noise = torch.normal(0, sigma * threshold, size=param.shape, device=param.device)
    param.data.add_(noise)
