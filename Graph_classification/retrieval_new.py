print("Importing...")
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from GCN import *
from Utils.math_distances import cosine_distance
from Utils.my_utils import *
from Utils.util import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import time
from train_utils import get_batch_data

torch.manual_seed(124)
np.random.seed(124)

print("Finished Importing")

torch.manual_seed(124)
np.random.seed(124)

print("Finished Importing")

print("Settings")

run_folder="../"
dataset = "Test_dataset"
STEP_dataset = "D:/3DStepGraphClassification/RapidDesign_Datasets/StepData/"
graphml_dataset = "D:/3DStepGraphClassification/RapidDesign_Datasets/GraphData/"
learning_rate=0.0005
batch_size=1
num_epochs=1
dropout=0.5
model_name = "GCN_model_05-31" # "Name of the model trained in train files"
model_path = "D:/3DStepGraphClassification/results/runs_GCN/GCN_model_05-31/Models/" + model_name

print("Using model at path:", model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# save paths
out_dir = "D:/3DStepGraphClassification/Output"
print("Results will be saved in:", out_dir)

print("Loading Graph data...")
use_degree_as_tag = False
fold = 0
graphs, num_classes = my_load_data(graphml_dataset, use_degree_as_tag)

train_graphs, test_graphs = separate_data(graphs, fold)
train_graphs, valid_graphs = split_data(train_graphs, perc=0.9)
print("# training graphs: ", len(train_graphs))
print_data_commposition(train_graphs)
print("# validation graphs: ", len(valid_graphs))
print_data_commposition(valid_graphs)
print("# test graphs: ", len(test_graphs))
print_data_commposition(test_graphs)
# Num of different STEP entities founded in the graph dataset
feature_dim_size = graphs[0].node_features.shape[1]
print("Loading data... finished!")



print("Creating model")

model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path))
children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1
output_layer = "attention"

class feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = model
        self.pretrained.eval()

        self.net = list(self.pretrained.children())[:]#-2
        self.pretrained = None

    def forward(self, adj, features):
        features = self.net[0](x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = self.net[1](x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = self.net[2](x=features, edge_index=adj)
        features = nn.functional.relu(features)
        scores = self.net[3](features)
        scores = torch.t(scores)

        scores = nn.functional.relu(self.net[4](scores))
        scores = self.net[5](scores)
        scores = F.log_softmax(scores, dim=1)
        return scores


retrieval_model = feature_extractor()
retrieval_model.eval()
print("Model loaded")


num_graphs = len(graphs)
# Get the size of the feature we are using
feat_size = output_shape = num_classes
# Preallocate the matrix for storing all the features
all_feats = np.zeros((num_graphs, feat_size))
times = []
with torch.no_grad():
    idx = np.arange(num_graphs)
    for i in range(0, len(graphs), batch_size):
        sampled_idx = idx[i:i + batch_size]
        if len(sampled_idx) == 0:
            continue
        batch_all_graphs = [graphs[j] for j in sampled_idx]
        all_X_concat, all_graph_labels, all_adj = get_batch_data(batch_all_graphs, device)
        start_time = time.time()
        features = retrieval_model(all_adj, all_X_concat)

        times.append(time.time()-start_time)

        all_feats[i] = np.array(features.cpu())
print(all_feats.shape)

print("Mean time:", np.mean(np.array(times)))


num_queries = len(test_graphs)
# Preallocate the matrix for storing all the features for the queries
query_feats = np.zeros((num_queries, feat_size))
i = 0
with torch.no_grad():
    idx = np.arange(num_queries)
    for i in range(0, len(test_graphs), batch_size):
        sampled_idx = idx[i:i + batch_size]
        if len(sampled_idx) == 0:
            continue
        batch_test_graphs = [test_graphs[j] for j in sampled_idx]
        test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
        features = retrieval_model(test_adj, test_X_concat)
        query_feats[i] = np.array(features.cpu())
print(query_feats.shape)



from sklearn.neighbors import NearestNeighbors

metric = "cosine"
nbrs = NearestNeighbors(n_neighbors=num_graphs, algorithm ='auto', metric=metric).fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)

print(distances.shape)
print(indices.shape)


#this function create a perfect ranking :)
def make_perfect_holidays_result(graphs, q_ids):
    perfect_idx =[]
    for qimno in q_ids:
        this_g = graphs[qimno]
        positive_results = set([i for i, gh in enumerate(graphs) if (gh.label == this_g.label)])
        ok=[qimno]+[i for i in  positive_results]
        others = [i for i in range(1491) if i not in positive_results and i != qimno]
        perfect_idx.append(ok+others)
    return np.array(perfect_idx)

def mAP(q_ids, idx, plot=False):
    aps = []
    precision_recall_x_class = {}
    for qimno, qres in zip(q_ids, idx):
        this_g = graphs[qimno]
        # collect the positive results in the dataset
        # the positives have the same prefix as the query image
        positive_results = set([i for i, gh in enumerate(graphs) if (gh.label == this_g.label)])
        #
        # ranks of positives. We skip the result #0, assumed to be the query image
        ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
        #
        # accumulate trapezoids with this basis
        recall_step = 1.0 / len(positive_results)
        ap = 0

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far
            # y-size on left side of trapezoid:
            precision_0 = ntp/float(rank) if rank > 0 else 1.0
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0

        aps.append(ap)

    return np.mean(aps)

query_imids = []
test_names = [g.name_graph for g in test_graphs]
for i, g in enumerate(graphs):
    if g.name_graph in test_names:
        query_imids.append(i)

perfect_result = make_perfect_holidays_result(graphs, query_imids)
p_map = mAP(query_imids,perfect_result)
print('Perfect mean AP = %.3f'%p_map)
map = mAP(query_imids, indices, True)
print('mean AP = %.3f'%map)


with open(out_dir + "/mAP_retrival.txt", 'a') as f:
    if isinstance(metric, str):
        metric_name = metric
    else:
        metric_name = metric.__name__
    f.write("Model: "+ str(model.__class__.__name__) + ", metric: "+ metric_name + ", out_layer dim:" + str(output_shape) + ", mAP: "+ str(map)+"\n")