import torch
from torch_geometric.data import DataLoader

from A_data_generator.uniform_lit_geometric_clause_generator import UniformLitGeometricClauseGenerator
from B_SAT_to_graph_converter.loader.dimac_loader import DimacLoader
from B_SAT_to_graph_converter.variable_to_variable_graph import VariableToVariableGraph
from C_GNNs.gcn_2_layer_linear_1_layer_gnn import GCN2LayerLinear1LayerGNN

from C_GNNs.nnconv_gnn import NNConvGNN
from D_trainer.adam_trainer import AdamTrainer
from F_visualiser.visualiser import Visualiser
from E_evaluator.model_evaluator import ModelEvaluator
from G_save.save_handler import SaveHandler
from collections import OrderedDict

#################################################
#
# CONSTANTS
#
#################################################

print("Starting the experiment")

experiment_configs = OrderedDict([
    # Generate data
    ("generator", UniformLitGeometricClauseGenerator(
        out_dir="../data_generated",
        percentage_sat=0.5,
        seed=None,
        min_n_vars=1,
        max_n_vars=5,
        min_n_clause=1,
        max_n_clause=5,
        lit_distr_p=0.4
    )),
    ("number_generated_data", 1000),

    # Load SATs and converting to graph data
    ("SAT_to_graph_converter", VariableToVariableGraph(
        max_clause_length=65
    )),
    ("percentage_training_set", 0.75),
    ("train_batch_size", 4),
    ("test_batch_size", 4),

    # Graph neural network structure
    ("gnn", GCN2LayerLinear1LayerGNN(
        sigmoid_output=True
    )),
    # ("gnn", NNConvGNN(
    #     sigmoid_output=True
    # )),

    # Train
    ("trainer", AdamTrainer(
        learning_rate=0.001,
        weight_decay=5e-4
    )),
    ("number_of_epochs", 100),

    # Eval


    # Visualise


    # Save

])

other_configs = {
# Generate data
"data_generated_folder_location": "../data_generated",

# Load SATs and converting to graph data


# Graph neural network structure


# Train


# Eval


# Visualise
"graph_directory_name": "../graphs",

# Save
"save_handler": SaveHandler,
"save_filename": "experiments.csv"

}


#################################################
#
# GENERATE SATS DATA
#
#################################################

print("\nGENERATING SATS DATA")

experiment_configs["generator"].delete_all()
experiment_configs["generator"].generate(experiment_configs["number_generated_data"])


#################################################
#
# LOAD SATS AND CONVERT TO GRAPH DATA
#
#################################################

print("\nLOADING SATS AND CONVERTING TO GRAPH DATA")

SAT_problems = DimacLoader().load_sat_problems()
dataset = experiment_configs["SAT_to_graph_converter"].convert_all(SAT_problems)

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
experiment_configs["gnn"].initialise_channels(
    next(iter(train_loader)).num_node_features,
    len(next(iter(train_loader)).y),
    next(iter(train_loader)).num_edge_features
)

model = experiment_configs["gnn"].to(device)


#################################################
#
# TRAIN
#
#################################################

print("\nTRAINING")

model_evaluator = ModelEvaluator(test_loader, device)
train_loss, test_loss, accuracy = experiment_configs["trainer"].train(
    experiment_configs["number_of_epochs"],
    model,
    train_loader,
    device,
    model_evaluator
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
# VISUALISE
#
#################################################

print("\nVISUALISE")

# Ask the user to save or not the results
save_user_input = ""
while save_user_input != "y" and save_user_input != "n":
    save_user_input = input("Save the results? (y or n)\n")
save_result = save_user_input == "y"

graph_filename = Visualiser().visualise(
    train_loss,
    test_loss,
    accuracy,
    other_configs["graph_directory_name"],
    save=save_result
)


#################################################
#
# SAVE
#
#################################################

print("\nSAVING")

if save_result:
    experiment_results = OrderedDict([
        ("test_loss", final_test_loss.item()),
        ("train_loss", train_loss[-1].item()),
        ("accuracy", final_accuracy),
        ("confusion_matrix", final_confusion_matrix),
        ("graph_filename", graph_filename)
    ])
    other_configs["save_handler"](experiment_configs, experiment_results, other_configs["save_filename"]).save()