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
from train_utils import get_batch_data, evaluate

torch.manual_seed(124)
np.random.seed(124)

print("Finished Importing")

torch.manual_seed(124)
np.random.seed(124)

print("Finished Importing")

print("Settings")

run_folder="../"
dataset = "MeviyData"
STEP_dataset = "D:/3DStepGraphClassification/MeviyData/TestData2/StepData/"
graphml_dataset = "D:/3DStepGraphClassification/MeviyData/TestData2/GraphData/"
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
out_dir = "D:/3DStepGraphClassification/Output_RPdata_3000"
print("Results will be saved in:", out_dir)

print("Loading Graph data...")
use_degree_as_tag = False
fold = 0
graphs, num_classes = my_load_data(graphml_dataset, use_degree_as_tag)

# train_graphs, test_graphs = separate_data(graphs, fold)
# train_graphs, valid_graphs = split_data(train_graphs, perc=0.9)
# print("# training graphs: ", len(train_graphs))
# print_data_commposition(train_graphs)
# print("# validation graphs: ", len(valid_graphs))
# print_data_commposition(valid_graphs)
# print("# test graphs: ", len(test_graphs))
# print_data_commposition(test_graphs)
# Num of different STEP entities founded in the graph dataset


feature_dim_size = graphs[0].node_features.shape[1]
# feature_dim_size = 71
num_classes = 10
print("Loading data... finished!")

print("Creating model")

# num_classes= 2
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

test_loss, test_acc, _ = evaluate(mmodel=retrieval_model, current_graphs=graphs, batch_size=batch_size, num_classes=num_classes, device=device, out_dir=out_dir, last_round=True)
print("Evaluate: loss on test: ", test_loss, " and accuracy: ", test_acc * 100)