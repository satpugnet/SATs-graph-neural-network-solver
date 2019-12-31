from collections import OrderedDict

import torch
from torch_geometric.data import DataLoader

from A_data_generator.data_generators.pigeon_hole_principle_generator import PigeonHolePrincipleGenerator
from A_data_generator.data_generators.uniform_lit_geometric_clause_generator import UniformLitGeometricClauseGenerator
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.clause_to_variable_graph import \
    ClauseToVariableGraph
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.variable_to_variable_graph import \
    VariableToVariableGraph
from C_GNNs.gnns.gcn_2_layer_linear_1_layer_gnn import GCN2LayerLinear1LayerGNN
from C_GNNs.gnns.nnconv_gnn import NNConvGNN
from C_GNNs.gnns.repeating_nnconv_gnn import RepeatingNNConvGNN
from D_trainer.trainers.adam_trainer import AdamTrainer
from E_evaluator.evaluators.default_evaluator import DefaultEvaluator
from F_visualiser.visualisers.visualiser import DefaultVisualiser
from G_save.save_handlers.save_handler import SaveHandler
from utils.dimac_loader import DimacLoader


# TODO: put this section in a different file
#################################################
#
# CONSTANTS
#
#################################################

print("Starting the experiment")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These configs will be saved to file
experiment_configs = OrderedDict([
    # Generate data
    ("generator", UniformLitGeometricClauseGenerator(
        out_dir="../data_generated",
        percentage_sat=0.5, # TODO: create a testing set where the percentage sat is not used (completly random sat problem)
        seed=None,
        min_n_vars=60,
        max_n_vars=80,
        min_n_clause=140,
        max_n_clause=180,
        lit_distr_p=0.4,
        include_trivial_clause=False
    )),
    # ("generator", PigeonHolePrincipleGenerator(
    #     out_dir="../data_generated",
    #     percentage_sat=0.5,
    #     seed=None,
    #     min_n_vars=None,
    #     max_n_vars=30,
    #     min_n_clause=None,
    #     max_n_clause=60,
    #     min_n_pigeons=1,
    #     max_n_pigeons=10,
    #     min_n_holes=1,
    #     max_n_holes=10
    # )),
    ("number_generated_data", 100),

    # Load SATs and converting to graph data
    # ("SAT_to_graph_converter", VariableToVariableGraph(
    #     max_clause_length=65
    # )),
    ("SAT_to_graph_converter", ClauseToVariableGraph()),
    ("percentage_training_set", 0.75),
    ("train_batch_size", 8),
    ("test_batch_size", 8),

    # Graph neural network structure
    ("gnn", GCN2LayerLinear1LayerGNN(
        sigmoid_output=True,
        dropout_prob=0.5
    )),
    # ("gnn", NNConvGNN(
    #     sigmoid_output=True,
    #     deep_nn=False,
    #     dropout_prob=0.5,
    #     num_hidden_neurons=24
    # )),
    # ("gnn", RepeatingNNConvGNN(
    #     sigmoid_output=True,
    #     dropout_prob=0,
    #     deep_nn=False,
    #     num_hidden_neurons=8,
    #     conv_repetition=10,
    #     ratio_test_train_rep=1
    # )),

    # Train
    ("trainer", AdamTrainer(
        learning_rate=0.001,
        weight_decay=5e-4,
        device=device,
        num_epoch_before_halving_lr=10
    )),
    ("number_of_epochs", 100),

    # Eval
    ("evaluator", DefaultEvaluator(
        device=device
    )),

    # Visualise


    # Save

])

# These configs will not be saved to file
other_configs = {
    # Generate data
    "data_generated_folder_location": "../data_generated",

    # Load SATs and converting to graph data


    # Graph neural network structure


    # Train


    # Eval


    # Visualise
    "visualiser": DefaultVisualiser(),
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

experiment_configs["evaluator"].test_loader = test_loader
train_loss, test_loss, accuracy, final_time = experiment_configs["trainer"].train(
    experiment_configs["number_of_epochs"],
    model,
    train_loader,
    experiment_configs["evaluator"]
)


#################################################
#
# EVAL
#
#################################################

print("\nEVALUATING")

model.eval()
final_test_loss, final_accuracy, final_confusion_matrix = experiment_configs["evaluator"].eval(model, do_print=True)


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

graph_filename = other_configs["visualiser"].visualise(
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
        ("time_taken", final_time),
        ("test_loss", final_test_loss.item()),
        ("train_loss", train_loss[-1].item()),
        ("accuracy", final_accuracy),
        ("confusion_matrix", final_confusion_matrix),
        ("graph_filename", graph_filename)
    ])
    other_configs["save_handler"](experiment_configs, experiment_results, other_configs["save_filename"]).save()