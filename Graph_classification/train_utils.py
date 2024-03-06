from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.util import *
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.nn as nn

from scipy.sparse import coo_matrix, block_diag
from torch_geometric.utils import to_dense_adj

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_labels = true_labels.type(torch.int64)
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl


def get_graphpool(batch_graph, device):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)

# def get_batch_data(batch_graph, device):
#     X_concat = np.concatenate([graph.node_features.cpu().numpy() for graph in batch_graph], 0)
#     X_concat = torch.from_numpy(X_concat).to(device)
    
#     adjj = np.concatenate([graph.edge_mat.cpu().numpy() for graph in batch_graph], 0)
#     adjj = torch.from_numpy(adjj).to(device, dtype=torch.int64)  # Ensure the dtype is int64
    
#     graph_labels = np.array([graph.label for graph in batch_graph])
#     graph_labels = torch.from_numpy(graph_labels).to(device)
    
#     return X_concat, graph_labels, adjj

def get_batch_data(batch_graph, device):
    # Create a list of node feature matrices and convert to tensors
    node_features_list = [torch.tensor(graph.node_features, dtype=torch.float).to(device) for graph in batch_graph]
    # Calculate the cumulative sum of nodes to adjust edge indices
    cumsum_nodes = np.cumsum([0] + [features.shape[0] for features in node_features_list])
    
    # Create a batched node feature matrix
    X_concat = torch.cat(node_features_list, dim=0)
    
    # Adjust edge indices for each graph and create a list of edge index tensors
    edge_indices_list = []
    for i, graph in enumerate(batch_graph):
        edge_indices = torch.tensor(graph.edge_mat, dtype=torch.long).to(device) + cumsum_nodes[i]
        edge_indices_list.append(edge_indices)

    # edge_indices_list = []
    # for i, graph in enumerate(batch_graph):
    #     edge_indices = torch.tensor(graph.edge_mat, dtype=torch.long).to(device) + cumsum_nodes[i]
    #     edge_indices_transposed = edge_indices.transpose(0, 1)  # Transpose to get [number_of_edges, 2] format
    #     edge_indices_list.append(edge_indices_transposed)
    
    for edge_indices in edge_indices_list:
        print(edge_indices.shape)  # Each should be [number_of_edges, 2]

    # Create a batched edge index tensor (sparse adjacency matrix representation)
    adjj = torch.cat(edge_indices_list, dim=1)
    
    # Create a tensor for graph labels
    graph_labels = torch.tensor([graph.label for graph in batch_graph], dtype=torch.long).to(device)
    
    return X_concat, graph_labels, adjj




# def get_batch_data(batch_graph, device):
#     X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
#     X_concat = torch.from_numpy(X_concat).to(device)
#     # graph-level sum pooling

#     adjj = np.concatenate([graph.edge_mat for graph in batch_graph], 0)
#     adjj = torch.from_numpy(adjj).to(device)

#     graph_labels = np.array([graph.label for graph in batch_graph])
#     graph_labels = torch.from_numpy(graph_labels).to(device)

#     return X_concat, graph_labels, adjj.to(torch.int64)


def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def load_graphml_files(batch_graph, selected_idx, train_graphs, device):
    loaded_graph_files = []
    for i, graph_file in enumerate(batch_graph):
        if isinstance(graph_file, S2VGraph):
            g_file = graph_file 
        else: 
            # g_file = my_load_single_graphml(graph_file, device=device)
             g_file = my_load_single_graphml(graph_file, device="cpu")

        # g_file = my_load_single_graphml(graph_file["file"], device=device)
        loaded_graph_files.append(g_file)
        train_graphs[selected_idx[i]] = g_file
        # print(g_file.node_features.shape[1])
    return loaded_graph_files

def load_validation_graph(valid_graphs, device):
    for i, graph_file in enumerate(valid_graphs):
        if isinstance(graph_file, S2VGraph):
            g_file = graph_file 
        else:
            g_file = my_load_single_graphml(graph_file, device=device)
        valid_graphs[i] = g_file
        # print(g_file.node_features.shape[1])
    return valid_graphs

def train(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    # Turn on the train mode
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        batch_graph = load_graphml_files(batch_graph, selected_idx, train_graphs, device=device)
        # load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        graph_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        # model probability scores
        print("adjj shape:", adjj.shape)

        prediction_scores = mmodel(adjj, X_concat)
        loss = cross_entropy(prediction_scores, graph_labels)
        torch.cuda.empty_cache()
        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()

        # Optional: Clear some memory
        torch.cuda.empty_cache()

def train_old(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    # Turn on the train mode
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        graph_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        # model probability scores
        prediction_scores = mmodel(adjj, X_concat)

        loss = cross_entropy(prediction_scores, graph_labels)
        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()

        
    # Optional: Clear some memory
        torch.cuda.empty_cache()

def extract_filenames(file_dicts):
    filenames = []
    for file_dict in file_dicts:
        full_path = file_dict['file']
        filename = os.path.basename(full_path)
        filenames.append(filename)
    return filenames

def train_new(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    
    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler()

    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        batch_graph = load_graphml_files(batch_graph, selected_idx, train_graphs, device=device)
        
        # Load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        graph_labels = label_smoothing(graph_labels, num_classes)
        
        optimizer.zero_grad()

        # Mixed precision context
        with autocast():
            prediction_scores = mmodel(adjj, X_concat)
            loss = cross_entropy(prediction_scores, graph_labels)

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # Clips grad norm
        scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)

        # optimizer's step is called in scaled update
        scaler.step(optimizer)
        scaler.update()

        # Optional: Clear some memory
        torch.cuda.empty_cache()


def predict_sat(mmodel, current_graphs, batch_size, num_classes, device):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            # model probability scores
            prediction_scores = mmodel(test_adj, test_X_concat)

            test_graph_labels = label_smoothing(test_graph_labels, num_classes)
            loss = cross_entropy(prediction_scores, test_graph_labels)
            total_loss += loss.item()
            prediction_output.append(prediction_scores.detach())

    # model probabilities output
    prediction_output = torch.cat(prediction_output, 0)
    # predicted labels
    predictions = prediction_output.max(1, keepdim=True)[1]
    # real labels
    labels = torch.LongTensor([graph.label for graph in current_graphs]).to(device)
    # num correct predictions
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    accuracy = correct / float(len(current_graphs))

def evaluate(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
   
    with torch.no_grad():
        # evaluating
        prediction_output = []
        prediction_graph_name = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            batch_test_graphs = load_graphml_files(batch_test_graphs, sampled_idx, current_graphs, device=device)
            prediction_graph_name.append(batch_test_graphs[0].name_graph)
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            print("test_adj:", test_adj.shape)
            # model probability scores
            prediction_scores = mmodel(test_adj, test_X_concat)
            print("prediction_scores:", prediction_scores.shape)

            test_graph_labels = label_smoothing(test_graph_labels, num_classes)
            loss = cross_entropy(prediction_scores, test_graph_labels)
            total_loss += loss.item()
            prediction_output.append(prediction_scores.detach())

    # model probabilities output
    prediction_output = torch.cat(prediction_output, 0)
    # predicted labels
    predictions = prediction_output.max(1, keepdim=True)[1]
    # code modified forｐrediction
    prediction_result = []
    for i,value in enumerate(predictions):
        prediction_result.append([prediction_graph_name[i] +' :: '+ str(value.tolist()[0])])
    # print(prediction_result)
    # real labels
    labels = torch.LongTensor([graph.label for graph in current_graphs]).to(device)

    print("predictions shape:", predictions.shape)
    print("labels shape:", labels.shape)

    correct = predictions.eq(labels).sum().cpu().item()

    # num correct predictions
    # correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    accuracy = correct / float(len(current_graphs))


    # confusion matrix and class accuracy
    matrix = confusion_matrix(np.array(labels.cpu()), np.array(predictions.cpu()))
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    acc_x_class = matrix.diagonal() * 100

    if last_round:
        # plot and save statistics
        print("Accuracy per class :")
        # print(acc_x_class)
        with open(out_dir + "/results.txt", 'w') as f:
            # code modified for prediction to write in file
            for item in prediction_result:
                f.write("%s\n" % str(item))

        ax = sns.heatmap(matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.savefig(out_dir + "/Confusion Matrix")

    return total_loss/len(current_graphs), accuracy, acc_x_class


# def evaluate(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False):
#     # Turn on the evaluation mode
#     mmodel.eval()
#     total_loss = 0.
#     prediction_output = []
#     prediction_graph_names = []
#     labels_list = []

#     with torch.no_grad():
#         # evaluating
#         idx = np.arange(len(current_graphs))
#         for i in range(0, len(current_graphs), batch_size):
#             sampled_idx = idx[i:i + batch_size]
#             if len(sampled_idx) == 0:
#                 continue
#             batch_test_graphs = [current_graphs[j] for j in sampled_idx]
#             batch_test_graphs = load_graphml_files(batch_test_graphs, sampled_idx, current_graphs, device=device)
            
#             for graph in batch_test_graphs:
#                 prediction_graph_names.append(graph.name_graph)
#                 labels_list.append(graph.label)
            
#             # load graph batch
#             test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
#             print("test_adj shape:", test_adj.shape)

#             test_graph_labels = label_smoothing(test_graph_labels, num_classes)
#             for adj in test_adj:
#                 prediction_scores = mmodel(adj, test_X_concat)
#                 loss = cross_entropy(prediction_scores, test_graph_labels)
#                 total_loss += loss.item()
#                 prediction_output.append(prediction_scores.detach())
            
#             # model probability scores
#             # prediction_scores = mmodel(test_adj, test_X_concat)
#             # print("prediction_scores shape:", prediction_scores.shape)
            
#             # test_graph_labels = label_smoothing(test_graph_labels, num_classes)
#             # loss = cross_entropy(prediction_scores, test_graph_labels)
#             # total_loss += loss.item()
#             # prediction_output.append(prediction_scores.detach())

#     # Concatenate all the prediction outputs and labels
#     prediction_output = torch.cat(prediction_output, 0)
#     # predicted labels
#     predictions = prediction_output.max(1, keepdim=True)[1]
#     labels = torch.LongTensor(labels_list).to(device).view(-1, 1)  # Ensure labels are correctly sized

#     correct = predictions.eq(labels).sum().cpu().item()

#     # Calculate accuracy
#     accuracy = correct / float(len(current_graphs))

#     # confusion matrix and class accuracy
#     matrix = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
#     matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#     acc_per_class = matrix_normalized.diagonal() * 100

#     if last_round:
#         # plot and save statistics
#         with open(f"{out_dir}/results.txt", 'w') as f:
#             for name, value in zip(prediction_graph_names, predictions.cpu().numpy()):
#                 f.write(f"{name} :: {value[0]}\n")

#         ax = sns.heatmap(matrix_normalized, annot=True, cmap='Blues')
#         ax.set_title('Confusion Matrix')
#         plt.savefig(f"{out_dir}/Confusion_Matrix.png")

#     return total_loss / len(current_graphs), accuracy, acc_per_class