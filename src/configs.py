# TODO: put this section in a different file
#################################################
#
# CONFIGS
#
#################################################
from collections import OrderedDict

import torch

from A_data_generator.data_generators.distr_based_generator.distr_based_generator_enum import Distribution
from A_data_generator.data_generators.distr_generator import DistrBasedGenerator
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.clause_to_variable_graph import \
    ClauseToVariableGraph
from C_GNNs.gnns.gcn_2_layer_linear_1_layer_gnn import GCN2LayerLinear1LayerGNN
from C_GNNs.gnns.nnconv_gnn import NNConvGNN
from C_GNNs.gnns.repeating_nnconv_gnn import RepeatingNNConvGNN
from C_GNNs.gnns.variable_repeating_nnconv_gnn import VariableRepeatingNNConvGNN
from D_trainer.trainers.adam_trainer import AdamTrainer
from E_evaluator.evaluators.default_evaluator import DefaultEvaluator
from F_visualiser.visualisers.visualiser import DefaultVisualiser
from G_save.save_handlers.save_handler import SaveHandler
from utils import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These configs will be saved to file
exp_configs = OrderedDict([
    # Generate data
    ("generator", DistrBasedGenerator(
        percentage_sat=0.5,
        seed=None,
        min_max_n_vars=(40, 60),
        min_max_n_clauses=(80, 140),
        var_num_distr=Distribution.UNIFORM,
        var_num_distr_params=[],
        clause_num_distr=Distribution.UNIFORM,
        clause_num_distr_params=[],
        lit_in_clause_distr=Distribution.GEOMETRIC,
        lit_in_clause_distr_params=[0.2],
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
    # ("test_generator", DistrBasedGenerator(
    #     percentage_sat=0.5,
    #     seed=None,
    #     min_max_n_vars=(10, 30),
    #     min_max_n_clauses=(30, 60),
    #     var_num_distr=Distribution.UNIFORM,
    #     var_num_distr_params=[],
    #     clause_num_distr=Distribution.UNIFORM,
    #     clause_num_distr_params=[],
    #     lit_in_clause_distr=Distribution.GEOMETRIC,
    #     lit_in_clause_distr_params=[0.4],
    #     include_trivial_clause=False
    # )),
    ("num_gen_data", 8000),
    ("percentage_training_set", 0.75),


    # Load SATs and converting to graph data
    # ("SAT_to_graph_converter", VariableToVariableGraph(
    #     max_clause_length=65
    # )),
    ("SAT_to_graph_converter", ClauseToVariableGraph()),
    ("train_batch_size", 32),
    ("test_batch_size", 32),


    # Graph neural network structure
    # ("gnn", GCN2LayerLinear1LayerGNN(
    #     sigmoid_output=True,
    #     dropout_prob=0.5
    # )),
    # ("gnn", NNConvGNN(
    #     sigmoid_output=True,
    #     deep_nn=False,
    #     dropout_prob=0.5,
    #     num_hidden_neurons=8
    # )),
    ("gnn", RepeatingNNConvGNN(
        sigmoid_output=True,
        dropout_prob=0,
        deep_nn=False,
        num_hidden_neurons=16,
        conv_repetition=20,
        ratio_test_train_rep=1
    )),
    # ("gnn", VariableRepeatingNNConvGNN(
    #     sigmoid_output=True,
    #     dropout_prob=0,
    #     deep_nn=False,
    #     num_hidden_neurons=32,
    #     conv_min_max_rep=(10, 15),
    #     ratio_test_train_rep=1
    # )),


    # Train
    ("trainer", AdamTrainer(
        learning_rate=0.001,
        weight_decay=5e-4,
        device=device,
        num_epoch_before_halving_lr=33
    )),
    ("number_of_epochs", 60),


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
