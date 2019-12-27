import os
from random import shuffle

import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
import numpy as np
from SAT_to_graph_converter.SAT_to_graph_converter import SATToGraphConverter
from data_generation.dimac_loader import DimacLoader

#################################################
#
# CONSTANTS
#
#################################################
from data_generation.dimacs_generators import DimacsGenerator
from eval.model_evaluator import ModelEvaluator

# Generate data
NUMBER_GENERATED_DATA = 1000
PERCENTAGE_SAT_IN_DATA = 0.5

# Load SATs and graph data
PERCENTAGE_TRAINING_SET = 0.75
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
MAX_CLAUSE_LENGTH = 15

# Graph neural network structure

# Train
NUMBER_OF_EPOCHS = 20

# Visualise
DATA_GENERATED_FOLDER_LOCATION = "../data_generated"

# Eval


#################################################
#
# GENERATE SATS DATA
#
#################################################

print("\nGENERATING SATS DATA")

generator = DimacsGenerator(DATA_GENERATED_FOLDER_LOCATION, percentage_sat=PERCENTAGE_SAT_IN_DATA)

generator.delete_all()
generator.generate(NUMBER_GENERATED_DATA)

#################################################
#
# LOAD SATS AND GRAPH DATA
#
#################################################

print("\nLOADING SATS AND GRAPH DATA")

SAT_problems = DimacLoader().load_sat_problems()

dataset = SATToGraphConverter(MAX_CLAUSE_LENGTH).convert_all(SAT_problems)
shuffle(dataset)

num_train = int(len(dataset) * PERCENTAGE_TRAINING_SET)

train_loader = DataLoader(dataset[:num_train], batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset[num_train:], batch_size=TEST_BATCH_SIZE, shuffle=False)

#################################################
#
# GRAPH NEURAL NETWORK STRUCTURE
#
#################################################

print("\nCREATING GRAPH NEURAL NETWORK STRUCTURE")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(next(iter(train_loader)).num_node_features, 16)
        self.conv2 = GCNConv(16, len(next(iter(train_loader)).y))

        self.fc2 = nn.Linear(len(next(iter(train_loader)).y), 1)

    # TODO: check how to take into account edge attribute
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        x = self.fc2(x)

        return torch.sigmoid(x)

#################################################
#
# TRAIN
#
#################################################

print("\nTRAINING")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# data = dataset[0].to(device)
# data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model_evaluator = ModelEvaluator(test_loader, device)

accuracy = []
test_loss = []
train_loss = []

model.train()
for epoch in range(NUMBER_OF_EPOCHS):
    print("Epoch: " + str(epoch))
    train_error = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y.view(-1, 1)) # F.nll_loss(out, batch.y)
        # print(out)
        # print(batch.y.view(-1, 1))
        # print(loss)
        train_error += loss

        loss.backward()
        optimizer.step()
    train_loss.append(train_error / len(train_loader))

    with torch.no_grad():
        current_test_loss, current_accuracy = model_evaluator.eval(model, train_loss[-1], do_print=True)
        test_loss.append(current_test_loss)
        accuracy.append(current_accuracy)

#################################################
#
# VISUALISE
#
#################################################

print("\nVISUALISE")

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

#################################################
#
# EVAL
#
#################################################

print("\nEVALUATING")

# model.eval()
# _, pred = model(data).max(dim=1)
# correct = float(pred.eq(data.y).sum().item())
# acc = correct / sum(data.y)
# print('Accuracy: {:.4f}'.format(acc))

model.eval()
current_test_loss, current_accuracy = model_evaluator.eval(model, TEST_BATCH_SIZE, do_print=True)

