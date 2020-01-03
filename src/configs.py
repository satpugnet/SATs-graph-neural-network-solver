# TODO: put this section in a different file
#################################################
#
# CONFIGS
#
#################################################
from collections import OrderedDict

import torch

from A_data_generator.data_generators.distr_based_generator import DistrBasedGenerator
from A_data_generator.data_generators.distr_based_generators.distr_based_generator_enum import Distribution
from B_SAT_to_graph_converter.SAT_to_graph_converters.clause_variable_graph_converter.clause_to_variable_graph import \
    ClauseToVariableGraph
from C_GNN.gnns.edge_atr_gnns_enums.aggr_enum import Aggr
from C_GNN.gnns.edge_attr_gnns.repeating_nnconv_gnns.variable_repeating_nnconv_gnn import VariableRepeatingNNConvGNN
from C_GNN.poolings.global_attention_pooling import GlobalAttentionPooling
from C_GNN.poolings.mean_pooling import MeanPooling
from C_GNN.poolings.set_to_set_pooling import SetToSetPooling
from C_GNN.poolings.sort_pooling import SortPooling
from D_trainer.trainers.adam_trainer import AdamTrainer
from E_evaluator.evaluators.default_evaluator import DefaultEvaluator
from F_visualiser.visualisers.visualiser import DefaultVisualiser
from G_save.save_handlers.save_handler import SaveHandler
from utils import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: add an option to put into a folder sample of (TP, FP, TN, FN) with the name similar to the graphs name for later visualisation
# TODO: Fix the bug when printing epoch percentages and print percentage for the testing
# TODO: replace this ordered dict by dict
# TODO: remove all the optional param everywhere
# TODO: create an all node graph so that the algorithms without edge_attr can be tested
# TODO: Create non-global pooling layer
# TODO: Check normalisation pooling layers
# These configs will be saved to a file when saving the experiment configurations, put important configs here
exp_configs = OrderedDict([
    # GENERATE SATS DATA
    ("generator", DistrBasedGenerator(  # The algorithm use to generate the SATs data
        percentage_sat=0.5,  # The percentage of SAT to UNSAT problems
        seed=None,  # The seed used if any
        min_max_n_vars=(20, 40),  # The min and max number of variable in the problems
        min_max_n_clauses=(50, 150),  # The min and max number of clauses in the problems
        var_num_distr=Distribution.UNIFORM,  # The distribution used to generate the number of variable in a problem
        var_num_distr_params=[],  # The distribution parameters
        clause_num_distr=Distribution.UNIFORM,  # The distribution used to generate the number of clauses in a problem
        clause_num_distr_params=[],  # The distribution parameters
        lit_in_clause_distr=Distribution.GEOMETRIC,  # The distribution used to generate the number of clauses in a problem
        lit_in_clause_distr_params=[0.2],  # The distribution parameters
        include_trivial_clause=False  # Whether to include clause containing a variable and its opposite such as (x and not x)
    )),
    # ("generator", PigeonHolePrincipleGenerator(
    #     percentage_sat=0.5,
    #     seed=None,
    #     min_max_n_vars=(None, 30),
    #     min_max_n_clauses=(None, 60),
    #     min_max_n_pigeons=(1, 10),  # The min and max number of pigeons
    #     min_max_n_holes=(1, 10),  # The min and max number of holes
    # )),
    ("test_generator", DistrBasedGenerator(  # (optional) The generator to use for the testing data, optional, if not set, the same distribution is used than the one for training
        percentage_sat=None,  # The percentage of SAT to UNSAT problems
        seed=None,  # The seed used if any
        min_max_n_vars=(20, 40),  # The min and max number of variable in the problems
        min_max_n_clauses=(50, 150),  # The min and max number of clauses in the problems
        var_num_distr=Distribution.UNIFORM,  # The distribution used to generate the number of variable in a problem
        var_num_distr_params=[],  # The distribution parameters
        clause_num_distr=Distribution.UNIFORM,  # The distribution used to generate the number of clauses in a problem
        clause_num_distr_params=[],  # The distribution parameters
        lit_in_clause_distr=Distribution.GEOMETRIC,  # The distribution used to generate the number of clauses in a problem
        lit_in_clause_distr_params=[0.2],  # The distribution parameters
        include_trivial_clause=False  # Whether to include clause containing a variable and its opposite such as (x and not x)
    )),
    ("num_gen_data", 4000),  # The amount of data to generate in total
    ("percentage_training_set", 0.75),  # The percentage of training data in total compare to testing


    # LOAD SATS AND CONVERT TO GRAPH DATA
    # ("SAT_to_graph_converter", VariableToVariableGraph(
    #     max_clause_length=65
    # )),
    ("SAT_to_graph_converter", ClauseToVariableGraph()),  # The algorithm used to convert from SAT problems to graph problems
    ("train_batch_size", 32),  # The size of the train batch
    ("test_batch_size", 32),  # The size of the test batch


    # GRAPH NEURAL NETWORK STRUCTURE
    # ("gnn", GCN2LayerLinear1LayerGNN(  # The GNN architecture to use
    #     sigmoid_output=True,  # Whether to output a sigmoid
    #     dropout_prob=0.5,  # The probability of dropout
    #     pooling=Pooling.GLOBAL_ADD,
    #     num_hidden_neurons=8
    # )),
    # ("gnn", NNConvGNN(
    #     sigmoid_output=True,
    #     deep_nn=False,  # Whether to use a deep neural net of shallow one
    #     pooling=Pooling.GLOBAL_ADD,
    #     dropout_prob=0.5,
    #     num_hidden_neurons=8,  # The number of hidden neurons in the hidden layers
    #     aggr=Aggr.ADD
    # )),
    # ("gnn", RepeatingNNConvGNN(
    #     sigmoid_output=True,
    #     dropout_prob=0,
    #     pooling=Pooling.GLOBAL_ADD,
    #     deep_nn=True,
    #     num_hidden_neurons=64,
    #     conv_repetition=20,  # The number of repetition of the ConvGNN
    #     ratio_test_train_rep=1,  # The ratio of the number of repetition of the ConvGNN for the testing and training
    #     aggr=Aggr.ADD,
    #     num_repeating_layer=1
    # )),
    ("gnn", VariableRepeatingNNConvGNN(
        sigmoid_output=True,
        dropout_prob=0,
        pooling=GlobalAttentionPooling(64, True),
        deep_nn=True,
        num_hidden_neurons=32,
        conv_min_max_rep=(10, 20),  # The range in which to uniformly pick for the number of repetition of the ConvGNN
        ratio_test_train_rep=2,
        aggr=Aggr.ADD,
        num_repeating_layer=3
    )),


    # TRAIN
    ("trainer", AdamTrainer(  # The trainer to use
        learning_rate=0.001,  # The learning rate
        weight_decay=5e-4,  # The weight decay
        device=device,  # The device used
        num_epoch_before_halving_lr=33,  # The number of epoch between each halving of the learning rate
        activate_amp=False
    )),
    ("number_of_epochs", 100),


    # EVAL
    ("evaluator", DefaultEvaluator(  # The default evaluator used
        device=device
    )),


    # VISUALISE


    # SAVE

])

# These configs will be saved to a file when saving the experiment configurations, put unimportant configs here
other_configs = {
    "debug": True,  # Whether to print the debug info or not
    "verbose": False,  # The verbosity of the printing

    # GENERATE SATS DATA
    "data_generated_train_folder_location": "../data_generated/train",  # Where to store the training data
    "data_generated_test_folder_location": "../data_generated/test",  # Where to store the testing data


    # LOAD SATS AND CONVERT TO GRAPH DATA


    # GRAPH NEURAL NETWORK STRUCTURE


    # TRAIN


    # EVAL


    # VISUALISE
    "visualiser": DefaultVisualiser(),  # The visualiser to display interesting information about the experiment
    "graph_directory_name": "../graphs",  # The location to store the graphs TODO: change name to plot


    # SAVE
    "save_handler": SaveHandler,  # The save handler for saving experiments
    "save_filename": "experiments.csv"  # The filename of the saved information
}

logger.init(debug=other_configs["debug"], verbose=other_configs["verbose"])
