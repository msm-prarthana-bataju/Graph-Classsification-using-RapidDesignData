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
from train_utils import *

torch.manual_seed(124)
np.random.seed(124)

print("Finished Importing")

print("Settings")

run_folder="../"
dataset = "RPDatasets"
STEP_dataset = "D:/3DStepGraphClassification_RapidDesignData/Datasets/RapidPrototype_TestData/StepData/"
graphml_dataset = "D:/3DStepGraphClassification_RapidDesignData/Datasets/RapidPrototype_TestData/GraphData/"
learning_rate=0.0005
batch_size=1
num_epochs=1
dropout=0.5
model_name = "GCN_model_02-16" # "Name of the model trained in train files"
model_path = "D:/3DStepGraphClassification_RapidDesignData/Datasets/RapidPrototype_TestData/GCN_model_02-16_100/Models/" + model_name

print("Using model at path:", model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# # save paths
# out_dir = "D:/3DStepGraphClassification_RapidDesignData/CosineOutput"
# print("Results will be saved in:", out_dir)


print("Loading Graph data...")
use_degree_as_tag = False
fold = 0
# graphs, num_classes, filenames = my_load_data_with_filename(graphml_dataset, use_degree_as_tag)
graphs, num_classes = my_load_data_without_graph_load(graphml_dataset)
filenames = extract_filenames(graphs)

train_graphs, test_graphs = separate_data_new(graphs, split_size=0.1)
train_graphs, valid_graphs = separate_data_new(train_graphs, split_size=0.1)
print("# training graphs: ", len(train_graphs))
# print_data_commposition(train_graphs)
print("# validation graphs: ", len(valid_graphs))
# print_data_commposition(valid_graphs)
print("# test graphs: ", len(test_graphs))
# print_data_commposition(test_graphs)

# feature_dim_size = graphs[0].node_features.shape[1]
feature_dim_size = 64
print("Loading data... finished!")

# train_graphs, test_graphs = separate_data(graphs, fold)
# train_graphs, valid_graphs = split_data(train_graphs, perc=0.9)
# print("# training graphs: ", len(train_graphs))
# print_data_commposition(train_graphs)
# print("# validation graphs: ", len(valid_graphs))
# print_data_commposition(valid_graphs)
# print("# test graphs: ", len(test_graphs))
# print_data_commposition(test_graphs)
# # Num of different STEP entities founded in the graph dataset
# feature_dim_size = graphs[0].node_features.shape[1]
# print(filenames)
# print("Loading data... finished!")


print("Creating model")

num_classes = 393
# model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=dropout).to(device)
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
        batch_all_graphs = load_graphml_files(batch_all_graphs, sampled_idx, graphs, device)
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
        batch_test_graphs = load_graphml_files(batch_test_graphs, sampled_idx, test_graphs, device)
        test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
        features = retrieval_model(test_adj, test_X_concat)
        query_feats[i] = np.array(features.cpu())
print(query_feats.shape)


from sklearn.neighbors import NearestNeighbors

metric = "cosine"
nbrs = NearestNeighbors(n_neighbors=num_graphs, algorithm ='auto', metric=metric).fit(all_feats)
distances, indices = nbrs.kneighbors(all_feats)

# Calculate the number of neighbors to retrieve (top 5)
num_neighbors = int(indices.shape[1] * 0.01)

# Open a text file to write the similarity scores
with open("similarity_scores_cosine.txt", "w") as file:
    # Iterate over each query sample
    for i, neighbors in enumerate(indices):
        print(f"Top {num_neighbors} neighbors for sample {i}:")
        top_neighbors = neighbors[:num_neighbors]
        top_distances = distances[i, :num_neighbors]
        for j, neighbor_index in enumerate(top_neighbors):
            filename = filenames[neighbor_index]
            similarity_score = 1 - top_distances[j]  # Calculate similarity score using cosine distance
            file.write(f"Index: {neighbor_index}, Similarity Score: {similarity_score:.4f}, Filename: {filename}\n")
        file.write("\n")
            # print(f"Index: {neighbor_index}, Distance: {top_distances[j]}")
            # print(f"Index: {neighbor_index}, Distance: {top_distances[j]}, Filename: {filename}")
        # print()
