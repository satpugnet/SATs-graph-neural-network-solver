import time
from collections import OrderedDict

import torch
from torch_geometric.data import DataLoader

from A_data_generator.data_generators.distr_based_generator.distr_based_generator_enum import Distribution
from A_data_generator.data_generators.distr_generator import DistrBasedGenerator
from A_data_generator.data_generators.pigeon_hole_principle_generator import PigeonHolePrincipleGenerator
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.clause_to_variable_graph import \
    ClauseToVariableGraph
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.variable_to_variable_graph import \
    VariableToVariableGraph
from C_GNNs.gnns.gcn_2_layer_linear_1_layer_gnn import GCN2LayerLinear1LayerGNN
from C_GNNs.gnns.nnconv_gnn import NNConvGNN
from C_GNNs.gnns.repeating_nnconv_gnn import RepeatingNNConvGNN
from C_GNNs.gnns.variable_repeating_nnconv_gnn import VariableRepeatingNNConvGNN
from D_trainer.trainers.adam_trainer import AdamTrainer
from E_evaluator.evaluators.default_evaluator import DefaultEvaluator
from F_visualiser.visualisers.visualiser import DefaultVisualiser
from G_save.save_handlers.save_handler import SaveHandler
from utils import logger
from utils.dimac_loader import DimacLoader
import logging

# TODO: put this section in a different file
#################################################
#
# CONSTANTS
#
#################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These configs will be saved to file
exp_configs = OrderedDict([
    # Generate data # TODO: generate and test from different sources
    ("generator", DistrBasedGenerator(
        percentage_sat=0.5, # TODO: create a testing set where the percentage sat is not used (completly random sat problem)
        seed=None,
        min_max_n_vars=(10, 30),
        min_max_n_clauses=(30, 60),
        var_num_distr=Distribution.UNIFORM,
        var_num_distr_params=[],
        clause_num_distr=Distribution.UNIFORM,
        clause_num_distr_params=[],
        lit_in_clause_distr=Distribution.GEOMETRIC,
        lit_in_clause_distr_params=[0.4],
        include_trivial_clause=False
    )),
    # ("generator", PigeonHolePrincipleGenerator(
    #     percentage_sat=0.5,
    #     seed=None,
    #     min_max_n_vars=(None, 30),
    #     min_max_n_clauses=(None, 60),
    #     min_max_n_pigeons=(1, 10),
    #     min_max_n_holes=(1, 10),
    # )),
    ("test_generator", DistrBasedGenerator(
        percentage_sat=0.5,
        seed=None,
        min_max_n_vars=(10, 30),
        min_max_n_clauses=(30, 60),
        var_num_distr=Distribution.UNIFORM,
        var_num_distr_params=[],
        clause_num_distr=Distribution.UNIFORM,
        clause_num_distr_params=[],
        lit_in_clause_distr=Distribution.GEOMETRIC,
        lit_in_clause_distr_params=[0.4],
        include_trivial_clause=False
    )),
    ("num_gen_data", 2000),
    ("percentage_training_set", 0.75),

    # Load SATs and converting to graph data
    # ("SAT_to_graph_converter", VariableToVariableGraph(
    #     max_clause_length=65
    # )),
    ("SAT_to_graph_converter", ClauseToVariableGraph()),
    ("train_batch_size", 16),
    ("test_batch_size", 16),

    # Graph neural network structure
    # ("gnn", GCN2LayerLinear1LayerGNN(
    #     sigmoid_output=True,
    #     dropout_prob=0.5
    # )),
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
    #     conv_repetition=2,
    #     ratio_test_train_rep=1
    # )),
    ("gnn", VariableRepeatingNNConvGNN(
        sigmoid_output=True,
        dropout_prob=0,
        deep_nn=False,
        num_hidden_neurons=8,
        conv_min_max_rep=(5, 10),
        ratio_test_train_rep=1
    )),

    # Train
    ("trainer", AdamTrainer(
        learning_rate=0.001,
        weight_decay=5e-4,
        device=device,
        num_epoch_before_halving_lr=33
    )),
    ("number_of_epochs", 10),

    # Eval
    ("evaluator", DefaultEvaluator(
        device=device
    )),

    # Visualise


    # Save

])

# These configs will not be saved to file
other_configs = {
    "debug": True,

    # Generate data
    "data_generated_train_folder_location": "../data_generated/train",
    "data_generated_test_folder_location": "../data_generated/test",

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

logger.init(debug=other_configs["debug"], verbose=False)


#################################################
#
# GENERATE SATS DATA
#
#################################################

logger.skip_line()
logger.get().info("GENERATING SATS DATA")

logger.get().info("Deleting old data")
exp_configs["generator"].delete_all(other_configs["data_generated_train_folder_location"])
exp_configs["generator"].delete_all(other_configs["data_generated_test_folder_location"])

logger.get().info("Generating data")
number_train_generated = int(exp_configs["num_gen_data"] * exp_configs["percentage_training_set"])
number_test_generated = exp_configs["num_gen_data"] - number_train_generated

if "test_generator" in exp_configs:
    exp_configs["test_generator"].generate(number_test_generated, other_configs["data_generated_test_folder_location"])
else:
    exp_configs["generator"].generate(number_test_generated, other_configs["data_generated_test_folder_location"])

exp_configs["generator"].generate(number_train_generated, other_configs["data_generated_train_folder_location"])


#################################################
#
# LOAD SATS AND CONVERT TO GRAPH DATA
#
#################################################

logger.skip_line()
logger.get().info("LOADING SATS AND CONVERTING TO GRAPH DATA")

train_SAT_problems = DimacLoader(other_configs["data_generated_train_folder_location"]).load_sat_problems()
test_SAT_problems = DimacLoader(other_configs["data_generated_test_folder_location"]).load_sat_problems()

train_dataset = exp_configs["SAT_to_graph_converter"].convert_all(train_SAT_problems)
test_dataset = exp_configs["SAT_to_graph_converter"].convert_all(test_SAT_problems)

train_loader = DataLoader(
    train_dataset,
    batch_size=exp_configs["train_batch_size"],
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=exp_configs["test_batch_size"],
    shuffle=True
)


#################################################
#
# GRAPH NEURAL NETWORK STRUCTURE
#
#################################################

logger.skip_line()
logger.get().info("CREATING GRAPH NEURAL NETWORK STRUCTURE")

exp_configs["gnn"].initialise_channels(
    next(iter(train_loader)).num_node_features,
    len(next(iter(train_loader)).y),
    next(iter(train_loader)).num_edge_features
)

model = exp_configs["gnn"].to(device)


#################################################
#
# TRAIN
#
#################################################

logger.skip_line()
logger.get().info("TRAINING")

exp_configs["evaluator"].test_loader = test_loader
train_loss, test_loss, accuracy, final_time = exp_configs["trainer"].train(
    exp_configs["number_of_epochs"],
    model,
    train_loader,
    exp_configs["evaluator"]
)


#################################################
#
# EVAL
#
#################################################

logger.skip_line()
logger.get().info("EVALUATING")

model.eval()
final_test_loss, final_accuracy, final_confusion_matrix = exp_configs["evaluator"].eval(model, do_print=True)


#################################################
#
# VISUALISE
#
#################################################

logger.skip_line()
logger.get().info("VISUALISE")

# Ask the user to save or not the results
save_user_input = ""
while save_user_input != "y" and save_user_input != "n":
    time.sleep(0.01) # Prevents problems of race condition with the logger
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

logger.skip_line()
logger.get().info("SAVING")

if save_result:
    experiment_results = OrderedDict([
        ("time_taken", final_time),
        ("test_loss", final_test_loss.item()),
        ("train_loss", train_loss[-1].item()),
        ("accuracy", final_accuracy),
        ("confusion_matrix", final_confusion_matrix),
        ("graph_filename", graph_filename)
    ])
    other_configs["save_handler"](exp_configs, experiment_results, other_configs["save_filename"]).save()