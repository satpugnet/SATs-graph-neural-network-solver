import torch
from torch_geometric.data import DataLoader

from GNNs.basic_gnn import BasicGNN
from SAT_to_graph_converter.SAT_to_graph_converter import SATToGraphConverter
from data_generation.dimac_loader import DimacLoader
from data_generation.dimacs_generators import DimacsGenerator

from eval.model_evaluator import ModelEvaluator
from training.model_trainer import ModelTrainer
from visualisation_handler.visualiser import Visualiser


#################################################
#
# CONSTANTS
#
#################################################

# Generate data
NUMBER_GENERATED_DATA = 10
PERCENTAGE_SAT_IN_DATA = 0.5

# Load SATs and converting to graph data
PERCENTAGE_TRAINING_SET = 0.75
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
MAX_CLAUSE_LENGTH = 15

# Graph neural network structure

# Train
NUMBER_OF_EPOCHS = 10

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
# LOAD SATS AND CONVERT TO GRAPH DATA
#
#################################################

print("\nLOADING SATS AND CONVERTING TO GRAPH DATA")

SAT_problems = DimacLoader().load_sat_problems()
dataset = SATToGraphConverter(MAX_CLAUSE_LENGTH).convert_all(SAT_problems)

num_train = int(len(dataset) * PERCENTAGE_TRAINING_SET)
train_loader = DataLoader(dataset[:num_train], batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset[num_train:], batch_size=TEST_BATCH_SIZE, shuffle=True)

#################################################
#
# GRAPH NEURAL NETWORK STRUCTURE
#
#################################################

print("\nCREATING GRAPH NEURAL NETWORK STRUCTURE")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BasicGNN(next(iter(train_loader)).num_node_features, len(next(iter(train_loader)).y)).to(device)

#################################################
#
# TRAIN
#
#################################################

print("\nTRAINING")

model_evaluator = ModelEvaluator(test_loader, device)
train_loss, test_loss, accuracy = ModelTrainer(model, train_loader, test_loader, device, model_evaluator).train(NUMBER_OF_EPOCHS)

#################################################
#
# VISUALISE
#
#################################################

print("\nVISUALISE")

Visualiser().visualise(train_loss, test_loss, accuracy)

#################################################
#
# EVAL
#
#################################################

print("\nEVALUATING")

model.eval()
current_test_loss, current_accuracy = model_evaluator.eval(model, TEST_BATCH_SIZE, do_print=True)

