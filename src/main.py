import time
from collections import OrderedDict

import torch
from torch_geometric.data import DataLoader

from configs import exp_configs, other_configs, device
from utils import logger
from utils.dimac_loader import DimacLoader


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
    int(len(next(iter(train_loader)).y) / exp_configs["train_batch_size"]),
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
train_loss, test_loss, accuracy, final_time, exp_configs["number_of_epochs"] = exp_configs["trainer"].train(
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
with torch.no_grad():
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
    save_user_input = input("\nSave the results? (y or n)\n")
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


logger.skip_line()
logger.get().info("EXPERIMENT COMPLETED\n")