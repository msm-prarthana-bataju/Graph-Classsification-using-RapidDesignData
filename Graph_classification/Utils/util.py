import os
import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import torch
import gc
import psutil

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class MyDataset(Dataset):
    def __init__(self, dataset_path, degree_as_tag=False):
        self.g_list = []
        self.label_dict = {}
        self.feat_dict = {}
        self.degree_as_tag = degree_as_tag

        for dir in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, dir)):
                print("Loading class:", dir)
                for file in os.listdir(os.path.join(dataset_path, dir)):
                    if file.endswith(".graphml"):
                        g = nx.read_graphml(os.path.join(dataset_path, dir, file))
                        l = dir
                        node_tags = []

                        if l not in self.label_dict:
                            mapped = len(self.label_dict)
                            self.label_dict[l] = mapped

                        for node in g:
                            node_lab = g.nodes[node]["type"]

                            if node_lab not in self.feat_dict:
                                mapped = len(self.feat_dict)
                                self.feat_dict[node_lab] = mapped

                            node_tags.append(self.feat_dict[node_lab])

                        self.g_list.append(S2VGraph(g, l, node_tags, name_graph=file))


    def __len__(self):
        return len(self.g_list)

    def __getitem__(self, idx):
        return self.g_list[idx]

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""

# class S2VGraph(object):
#     def __init__(self, g, label, node_tags=None, node_features=None, name_graph=None):
#         '''
#             g: a networkx graph
#             label: an integer graph label
#             node_tags: a list of integer node tags
#             node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
#             edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
#             neighbors: list of neighbors (without self-loop)
#         '''
#         self.label = label
#         self.g = g
#         self.node_tags = node_tags
#         self.neighbors = []
#         self.node_features = 0
#         self.edge_mat = 0
#         self.max_neighbor = 0
#         self.name_graph = name_graph
class S2VGraph:
    # Assuming this is a defined class elsewhere in your code
    def __init__(self, g, label, node_tags, name_graph):
        self.g = g
        self.label = label
        self.node_tags = node_tags
        self.name_graph = name_graph
        # Initialize additional properties to be filled later
        self.neighbors = []
        self.edge_mat = None
        self.node_features = None
        self.max_neighbor = 0

# def my_load_data(dataset, degree_as_tag=False):
#     g_list = []
#     label_dict = {}
#     feat_dict = {}
#     dataset_path = dataset
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for dir in os.listdir(dataset_path):
#         if os.path.isdir(dataset_path + dir):
#             print("Loading class:", dir)
#             for file in os.listdir(dataset_path + dir + "/"):
#                 if file.endswith(".graphml"):
#                     g = nx.read_graphml(dataset_path + dir + "/" + file)
#                     l = dir
#                     node_tags = []
#                     if l not in label_dict:
#                         mapped = len(label_dict)
#                         label_dict[l] = mapped
#                     for node in g:
#                         node_lab = g.nodes[node]["type"]
#                         if node_lab not in feat_dict:
#                             mapped = len(feat_dict)
#                             feat_dict[node_lab] = mapped
#                         node_tags.append(feat_dict[node_lab])
#                     g_list.append(S2VGraph(g, l, node_tags, name_graph=file))
#             # Free up memory after processing each directory
#             gc.collect()

#     # Process graphs
#     for g in g_list:
#         process_graph(g, label_dict)

#     if degree_as_tag:
#         for g in g_list:
#             g.node_tags = list(dict(g.g.degree).values())

#     # Extracting unique tag labels
#     tagset = set(tag for g in g_list for tag in g.node_tags)
#     tag2index = {tag: i for i, tag in enumerate(tagset)}

#     for g in g_list:
#         g.node_features = torch.zeros((len(g.node_tags), len(tagset)), dtype=torch.float32, device=device)
#         g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

#     print('# classes: %d' % len(label_dict))
#     print('# maximum node tag: %d' % len(tagset))
#     print("# data: %d" % len(g_list))

#     return g_list, len(label_dict)

def clear_memory():
    """Force garbage collection to free up memory."""
    gc.collect()
    print("Memory cleared. Current usage:", psutil.virtual_memory().percent, "%")



def my_load_data_without_graph_load(dataset):
    graphml_files = [] # List to store found .graphml files
    subfolder_count = 0  # Counter for subfolders
    
    # Walk through the directory tree
    label = 0
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith(".graphml"):
                g = {}
                g["file"] = os.path.join(root, file)
                g["label"] = label - 1
                graphml_files.append(g)
        
        # Count subfolders
        subfolder_count += len(dirs)
        label += 1

    return graphml_files, subfolder_count


# def my_load_single_graphml_new(dataset, device, degree_as_tag=False):
#     g_list = []
#     label_dict = {}
#     feat_dict = {}
#     dataset_path = dataset
#     # Carico un grafo, il valore dei nodi è il loro tipo
#     if dataset.endswith(".graphml"):
#         g = nx.read_graphml(dataset)
#         l = dir
#         node_tags = []
#         if not l in label_dict:
#             mapped = len(label_dict)
#             label_dict[l] = mapped
#         for node in g:
#             node_lab = g.nodes[node]["type"]
#             if not node_lab in feat_dict:
#                 mapped = len(feat_dict)
#                 feat_dict[node_lab] = mapped
#             node_tags.append(feat_dict[node_lab])
#         g_list.append(S2VGraph(g, l, node_tags, name_graph=os.path.basename(dataset)))
#     # add labels and edge_mat
#     for g in g_list:
#         # i miei grafi hanno id in stringa del tipo "#1", qui li vuole in int. Li converto.
#         dict_node_id = {}
#         for node in g.g:
#             idx = node
#             if not idx in dict_node_id:
#                 mapped = len(dict_node_id)
#                 dict_node_id[idx] = mapped

#         g.neighbors = [[] for i in range(len(g.g))]
#         for i, j in g.g.edges():
#             int_i = dict_node_id[i]
#             int_j = dict_node_id[j]
#             g.neighbors[int_i].append(int_j)
#             g.neighbors[int_j].append(int_i)
#         degree_list = []
#         for i in range(len(g.g)):
#             g.neighbors[i] = g.neighbors[i]
#             degree_list.append(len(g.neighbors[i]))
#         g.max_neighbor = max(degree_list)

#         g.label = label_dict[g.label]

#         edges = []
#         for pair in g.g.edges():
#             g1, g2 = pair
#             edges.append([dict_node_id[g1], dict_node_id[g2]])
#         edges.extend([[i, j] for j, i in edges])
#         deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
#         g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0))

#     if degree_as_tag:
#         for g in g_list:
#             g.node_tags = list(dict(g.g.degree).values())

#     # Extracting unique tag labels
#     tagset = set([])
#     for g in g_list:
#         tagset = tagset.union(set(g.node_tags))

#     tagset = list(tagset)
#     tag2index = {tagset[i]: i for i in range(len(tagset))}

#     for g in g_list:
#         g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
#         g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

#     print('# classes: %d' % len(label_dict))
#     print('# maximum node tag: %d' % len(tagset))

#     print("# data: %d" % len(g_list))

#     return g_list[0]

def my_load_single_graphml(file, device,  degree_as_tag=False):
    g_list = []
    label_dict = {}
    feat_dict = {}
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = file["file"]
    # Load a single GraphML file
    if file_path.endswith(".graphml"):
        g = nx.read_graphml(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))
        label = dir_name
        node_tags = [feat_dict.setdefault(g.nodes[node]["type"], len(feat_dict)) for node in g]
        g_list.append(S2VGraph(g, file["label"], node_tags, name_graph=os.path.basename(file_path)))

    # Process the loaded graph
    for g in g_list:
        process_graph(g, file["label"], degree_as_tag, device)
        del g.g  # Remove the original graph to free memory
        clear_memory()

    feat_dim = g_list[0].node_features.shape[1]
    return g_list[0]


def my_load_data(dataset, degree_as_tag=False):
    g_list = []
    label_dict = {}
    feat_dict = {}
    dataset_path = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dir_name in sorted(os.listdir(dataset_path)):
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.isdir(dir_path):
            # Clear memory before loading new data
            clear_memory()
            print("Loading class:", dir_name)
            for file_name in sorted(os.listdir(dir_path)):
                if file_name.endswith(".graphml"):
                    file_path = os.path.join(dir_path, file_name)
                    g = nx.read_graphml(file_path)
                    label = dir_name
                    node_tags = [feat_dict.setdefault(g.nodes[node]["type"], len(feat_dict)) for node in g]
                    g_list.append(S2VGraph(g, label, node_tags, name_graph=file_name))
                    # Optionally clear memory here if each file is large
                    clear_memory()

    # Optionally process graphs in batches to reduce memory usage
    for g in g_list:
        process_graph(g, label_dict, degree_as_tag, device)
        # Clear memory after processing each graph to ensure minimal memory footprint
        del g.g  # Remove the original graph to free memory
        clear_memory()

    print('# classes:', len(label_dict))
    print('# maximum node tag:', len(feat_dict))
    print("# data:", len(g_list))

    return g_list, len(label_dict)

def process_graph(g, label_dict, degree_as_tag, device):
    # label_dict.setdefault(g.label, len(label_dict))
    g.label = label_dict

    # Convert string node IDs to consecutive integers
    id_map = {node: i for i, node in enumerate(g.g.nodes())}
    edges = [[id_map[i], id_map[j]] for i, j in g.g.edges()]
    edges.extend([[j, i] for i, j in edges])  # For undirected graph symmetry

    g.edge_mat = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    if degree_as_tag:
        g.node_tags = [g.g.degree(node) for node in g.g.nodes()]

    # Fixed-size node features one-hot encoding
    node_features = torch.zeros(len(g.node_tags), 64, dtype=torch.float32, device=device) # Fixed dimension of 64
    for i, tag in enumerate(g.node_tags):
        index = tag - 1 if tag <= 64 else 0  # Map tag to index, use 0 for tags > 64 or when using 1 as fallback
        node_features[i, index] = 1
    g.node_features = node_features

    check_feature_dim_size = g.node_features.shape[1]
    # Now, check_feature_dim_size should always be 64

    # Clear local variables that are no longer needed to free memory
    del id_map, edges, node_features
    gc.collect()  # Explicitly collect garbage


# def process_graph(g, label_dict, degree_as_tag, device):
#     label_dict.setdefault(g.label, len(label_dict))
#     g.label = label_dict[g.label]

#     # Convert string node IDs to consecutive integers
#     id_map = {node: i for i, node in enumerate(g.g.nodes())}
#     edges = [[id_map[i], id_map[j]] for i, j in g.g.edges()]
#     edges.extend([[j, i] for i, j in edges])  # Assume undirected graph for symmetry

#     g.edge_mat = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

#     if degree_as_tag:
#         g.node_tags = [g.g.degree[node] for node in g.g.nodes()]

#     # Node features one-hot encoding
#     tagset = list(set(g.node_tags))
#     tag2index = {tag: i for i, tag in enumerate(tagset)}
#     node_features = torch.zeros(len(g.node_tags), len(tagset), dtype=torch.float32, device=device)
#     node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
#     g.node_features = node_features

#     check_feature_dim_size = g.node_features.shape[1]

#     # Clear local variables that are no longer needed to free memory
#     del id_map, edges, node_features
#     gc.collect()  # Explicitly collect garbage
    
# def process_graph(g, label_dict):
#     """
#     Function to process each graph and construct necessary attributes
#     """
#     dict_node_id = {node: idx for idx, node in enumerate(g.g.nodes())}

#     g.neighbors = [[] for _ in g.g.nodes()]
#     for i, j in g.g.edges():
#         g.neighbors[dict_node_id[i]].append(dict_node_id[j])
#         g.neighbors[dict_node_id[j]].append(dict_node_id[i])

#     degree_list = [len(neighbors) for neighbors in g.neighbors]
#     g.max_neighbor = max(degree_list)

#     g.label = label_dict[g.label]

#     edges = [[dict_node_id[i], dict_node_id[j]] for i, j in g.g.edges()]
#     edges.extend([[j, i] for i, j in edges])
#     g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0))

#     # Explicitly free up memory
#     del dict_node_id, degree_list, edges
#     gc.collect()

    
# def my_load_data(dataset, degree_as_tag=False):
#     g_list = []
#     label_dict = {}
#     feat_dict = {}
#     dataset_path = dataset
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Carico un grafo, il valore dei nodi è il loro tipo
#     for dir in os.listdir(dataset_path):
#         if os.path.isdir(dataset_path + dir):
#             print("Loading class:", dir)
#             for file in os.listdir(dataset_path + dir + "/"):
#                 if file.endswith(".graphml"):
#                     g = nx.read_graphml(dataset_path + dir + "/" + file)
#                     l = dir
#                     node_tags = []
#                     if not l in label_dict:
#                         mapped = len(label_dict)
#                         label_dict[l] = mapped
#                     for node in g:
#                         node_lab = g.nodes[node]["type"]
#                         if not node_lab in feat_dict:
#                             mapped = len(feat_dict)
#                             feat_dict[node_lab] = mapped
#                         node_tags.append(feat_dict[node_lab])
#                     g_list.append(S2VGraph(g, l, node_tags, name_graph=file))
#             gc.collect()
#     # add labels and edge_mat
#     for g in g_list:
#         # i miei grafi hanno id in stringa del tipo "#1", qui li vuole in int. Li converto.
#         dict_node_id = {}
#         for node in g.g:
#             idx = node
#             if not idx in dict_node_id:
#                 mapped = len(dict_node_id)
#                 dict_node_id[idx] = mapped

#         g.neighbors = [[] for i in range(len(g.g))]
#         for i, j in g.g.edges():
#             int_i = dict_node_id[i]
#             int_j = dict_node_id[j]
#             g.neighbors[int_i].append(int_j)
#             g.neighbors[int_j].append(int_i)
#         degree_list = []
#         for i in range(len(g.g)):
#             g.neighbors[i] = g.neighbors[i]
#             degree_list.append(len(g.neighbors[i]))
#         g.max_neighbor = max(degree_list)

#         g.label = label_dict[g.label]

#         edges = []
#         for pair in g.g.edges():
#             g1, g2 = pair
#             edges.append([dict_node_id[g1], dict_node_id[g2]])
#         edges.extend([[i, j] for j, i in edges])
#         deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
#         g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0))

#     if degree_as_tag:
#         for g in g_list:
#             g.node_tags = list(dict(g.g.degree).values())

#     # Extracting unique tag labels
#     tagset = set([])
#     for g in g_list:
#         tagset = tagset.union(set(g.node_tags))

#     tagset = list(tagset)
#     tag2index = {tagset[i]: i for i in range(len(tagset))}

#     for g in g_list:
#         g.node_features = torch.zeros((len(g.node_tags), len(tagset)), dtype=torch.float32, device=device)
#         g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

#     print('# classes: %d' % len(label_dict))
#     print('# maximum node tag: %d' % len(tagset))

#     print("# data: %d"% len(g_list))

#     return g_list, len(label_dict) 

def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('../dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            # n è il numero di nodi seguente
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            # per ogni nodo j
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            # g è il grafo, l è la classe del grafo, node_tags è una lista in cui per ogni nodo c'è l'attributo
            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def separate_data_new(data_list, split_size=0.2, random_state=5):
    labels = [entry['label'] for entry in data_list]
    files = [entry['file'] for entry in data_list]

    # Split the data into train and test sets
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=split_size, random_state=random_state, stratify=labels
    )

    # Create train and test data dictionaries
    train_data = [{'file': file, 'label': label} for file, label in zip(train_files, train_labels)]
    test_data = [{'file': file, 'label': label} for file, label in zip(test_files, test_labels)]

    return train_data, test_data


def separate_data(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

"""Get indexes of train and test sets"""
def separate_data_idx(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx






