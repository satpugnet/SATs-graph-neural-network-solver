import torch
from torch_geometric.data import DataLoader

from A_data_generator.uniform_lit_geometric_clause_generator import UniformLitGeometricClauseGenerator
from B_SAT_to_graph_converter.loader.dimac_loader import DimacLoader

from B_SAT_to_graph_converter.variable_to_variable_graph import VariableToVariableGraph
from C_GNNs.basic_gnn import BasicGNN
from D_trainer.adam_trainer import AdamTrainer

from E_visualiser.visualiser import Visualiser
from F_evaluator.model_evaluator import ModelEvaluator
from G_save.save_handler import SaveHandler


#################################################
#
# CONSTANTS
#
#################################################

print("Starting the experiment")

experiment_configs = {
# Generate data
"generator": UniformLitGeometricClauseGenerator,
"number_generated_data": 10,
"percentage_sat_in_data": 0.5,
"min_n_vars": 10,
"max_n_vars": 30,
"min_n_clause": 30,
"max_n_clause": 60,
"list_distr_p": 0.4,
"seed": None,

# Load SATs and converting to graph data
"SAT_to_graph_converter": VariableToVariableGraph,
"percentage_training_set": 0.75,
"train_batch_size": 4,
"test_batch_size": 4,
"max_clause_length": 65,

# Graph neural network structure
"gnn": BasicGNN,

# Train
"trainer": AdamTrainer,
"number_of_epochs": 10,
"learning_rate": 0.001,
"weight_decay": 5e-4,

# Visualise


# Eval


# Save

}

other_configs = {
# Generate data


# Load SATs and converting to graph data


# Graph neural network structure


# Train


# Visualise
"data_generated_folder_location": "../data_generated",
"graph_directory_name": "../graphs",

# Eval


# Save
"save_handler": SaveHandler,

}




#################################################
#
# GENERATE SATS DATA
#
#################################################

print("\nGENERATING SATS DATA")

# TODO: Fix the fact that we can only do a specific generator because it takes parameters
generator = experiment_configs["generator"](
    other_configs["data_generated_folder_location"],
    percentage_sat=experiment_configs["percentage_sat_in_data"],
    seed=experiment_configs["seed"],
    min_n_vars=experiment_configs["min_n_vars"],
    max_n_vars=experiment_configs["max_n_vars"],
    min_n_clause=experiment_configs["min_n_clause"],
    max_n_clause=experiment_configs["max_n_clause"],
    lit_distr_p=experiment_configs["list_distr_p"]
)

generator.delete_all()
generator.generate(experiment_configs["number_generated_data"])


#################################################
#
# LOAD SATS AND CONVERT TO GRAPH DATA
#
#################################################

print("\nLOADING SATS AND CONVERTING TO GRAPH DATA")

SAT_problems = DimacLoader().load_sat_problems()
dataset = experiment_configs["SAT_to_graph_converter"](
    experiment_configs["max_clause_length"]
).convert_all(SAT_problems)

num_train = int(len(dataset) * experiment_configs["percentage_training_set"])
train_loader = DataLoader(
    dataset[:num_train],
    batch_size=experiment_configs["train_batch_size"],
    shuffle=True
)
test_loader = DataLoader(
    dataset[num_train:],
    batch_size=experiment_configs["test_batch_size"],
    shuffle=True
)


#################################################
#
# GRAPH NEURAL NETWORK STRUCTURE
#
#################################################

print("\nCREATING GRAPH NEURAL NETWORK STRUCTURE")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = experiment_configs["gnn"](
    next(iter(train_loader)).num_node_features,
    len(next(iter(train_loader)).y)
).to(device)


#################################################
#
# TRAIN
#
#################################################

print("\nTRAINING")

model_evaluator = ModelEvaluator(test_loader, device)
trainer = experiment_configs["trainer"](
    model,
    train_loader,
    test_loader,
    device,
    model_evaluator,
    experiment_configs["learning_rate"],
    experiment_configs["weight_decay"]
)
train_loss, test_loss, accuracy = trainer.train(experiment_configs["number_of_epochs"])


#################################################
#
# VISUALISE
#
#################################################

print("\nVISUALISE")

graph_filename = Visualiser().visualise(
    train_loss,
    test_loss,
    accuracy,
    other_configs["graph_directory_name"]
)


#################################################
#
# EVAL
#
#################################################

print("\nEVALUATING")

model.eval()
final_test_loss, final_accuracy, final_confusion_matrix = model_evaluator.eval(model, do_print=True)


#################################################
#
# SAVE
#
#################################################

print("\nSAVING")

experiment_results = {
    "test_loss": final_test_loss.item(),
    "train_loss": train_loss[-1].item(),
    "accuracy": final_accuracy,
    "confusion_matrix": final_confusion_matrix,
    "graph_filename": graph_filename
}
other_configs["save_handler"](experiment_results, experiment_configs).save()