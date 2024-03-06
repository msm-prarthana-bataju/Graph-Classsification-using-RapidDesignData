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

print("Settings")

run_folder="../"
dataset = "MeviyData"
STEP_dataset = "D:/3DStepGraphClassification/MeviyData/RapidPrototype_3000/StepData/"
graphml_dataset = "D:/3DStepGraphClassification/MeviyData/RapidPrototype_3000/GraphData/"
learning_rate=0.0005
batch_size=1
num_epochs=1
dropout=0.5
model_name = "GCN_model_06-02" # "Name of the model trained in train files"
model_path = "D:/3DStepGraphClassification/RapidDesign_Results/runs_GCN/GCN_model_06-02/Models/" + model_name

print("Using model at path:", model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# save paths
out_dir = "D:/3DStepGraphClassification/Output_Meviy"
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
print(f"feature_dim_size: {feature_dim_size}")
print("Loading data... finished!")


def load_state_dict_ignore_scoring_layer(model, model_path):
    state_dict = torch.load(model_path)
    state_dict = {k: v for k, v in state_dict.items() if 'scoring_layer' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model


    print("Creating model")

# model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=dropout).to(device)
# model.load_state_dict(torch.load(model_path))

# Load the model, ignoring the weights for the scoring layer
model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=dropout).to(device)
model = load_state_dict_ignore_scoring_layer(model, model_path)

children_counter = 0
for n, c in model.named_children():
    print("Children Counter:", children_counter, "Layer Name:", n)
    children_counter += 1
output_layer = "attention"

class feature_extractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pretrained = model
        self.pretrained.eval()

        self.net = list(self.pretrained.children())[:-2] # Remove last two layers (fully_connected_first, scoring_layer)

    def forward(self, adj, features):
        for layer in self.net:
            if isinstance(layer, GCNConv):
                features = layer(features, adj)
                features = nn.functional.relu(features)
            else:
                features = layer(features)
        return features


retrieval_model = feature_extractor(model)
retrieval_model.eval()

num_data_points = len(graphs) # You should set this to the total number of data points you have

# Preallocate the matrix for storing all the features
all_feats = np.zeros((num_data_points, 32))  # Changed from num_classes to 32

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

        # Assume features are batched. If they're not, you should modify this line
        all_feats[i:i + batch_size] = np.array(features.cpu()).reshape(batch_size, -1)

np.savetxt('featurevector_rpdata_3000.txt', all_feats, delimiter=' ')


# # Preallocate the matrix for storing all the features
# all_feats = np.zeros((num_graphs, 32))  # Changed from num_classes to 32

# with torch.no_grad():
#     idx = np.arange(num_graphs)
#     for i in range(0, len(graphs), batch_size):
#         sampled_idx = idx[i:i + batch_size]
#         if len(sampled_idx) == 0:
#             continue
#         batch_all_graphs = [graphs[j] for j in sampled_idx]
#         all_X_concat, all_graph_labels, all_adj = get_batch_data(batch_all_graphs, device)
#         start_time = time.time()
#         features = retrieval_model(all_adj, all_X_concat)

#         times.append(time.time()-start_time)

#         all_feats[i] = np.array(features.cpu()).reshape(-1)
#         # all_feats[i] = np.array(features.cpu())
#         print(all_feats[i])

# np.savetxt('featurevectors_rapidprototype1.txt', all_feats, delimiter=' ')

