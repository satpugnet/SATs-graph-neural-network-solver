import time

import torch
from torch_geometric.data import DataListLoader

from C_GNN.data_parallel_wrapper import DataParallelWrapper
from configs import exp_configs, other_configs
from utils import logger
from utils.dimac_loader import DimacLoader
from torch_geometric.data import DataLoader


#################################################
#
# GENERATE SATS DATA
#
#################################################
from utils.multi_gpu import MultiGpu
from utils.user_input_querier import UserInputQuerier

logger.skip_line()
logger.get().info("GENERATING SATS DATA")

number_train_generated = int(exp_configs["num_gen_data"] * exp_configs["percentage_training_set"])
number_test_generated = exp_configs["num_gen_data"] - number_train_generated

regenerate_train_data = not other_configs["ask_for_regenerating_data"] or UserInputQuerier.ask("Regenerate train data?")
regenerate_test_data = not other_configs["ask_for_regenerating_data"] or UserInputQuerier.ask("Regenerate test data?")

generation_start_time = time.time()
if regenerate_train_data:
    logger.get().info("Generating training data")
    exp_configs["generator"].delete_all(other_configs["data_generated_train_folder_location"])
    exp_configs["generator"].generate(number_train_generated, other_configs["data_generated_train_folder_location"])

train_generation_completed_time = time.time()
if regenerate_test_data:
    logger.get().info("Generating testing data")
    exp_configs["generator"].delete_all(other_configs["data_generated_test_folder_location"])
    if "test_generator" in exp_configs:
        exp_configs["test_generator"].generate(number_test_generated, other_configs["data_generated_test_folder_location"])
    else:
        exp_configs["generator"].generate(number_test_generated, other_configs["data_generated_test_folder_location"])

logger.get().info("The generation of the data took a total of {:.1f}s ({:.1f}s for training data and {:.1f}s for testing data)"
                  .format(time.time() - generation_start_time, train_generation_completed_time - generation_start_time, time.time() - train_generation_completed_time))

#################################################
#
# LOAD SATS AND CONVERT TO GRAPH DATA
#
#################################################

logger.skip_line()
logger.get().info("LOADING SATS AND CONVERTING TO GRAPH DATA")

logger.get().info("Loading training data")
train_SAT_problems = DimacLoader(other_configs["data_generated_train_folder_location"]).load_sat_problems()
logger.get().info("Loading testing data")
test_SAT_problems = DimacLoader(other_configs["data_generated_test_folder_location"]).load_sat_problems()

logger.get().info("Converting train data to graph data")
train_dataset = exp_configs["SAT_to_graph_converter"].convert_all(train_SAT_problems)
logger.get().info("Converting test data to graph data")
test_dataset = exp_configs["SAT_to_graph_converter"].convert_all(test_SAT_problems)

logger.get().info("Loading the training data")
Loader = DataListLoader if MultiGpu.is_enabled() else DataLoader

if exp_configs["test_batch_size"] >= len(test_dataset) or exp_configs["train_batch_size"] >= len(train_dataset):
    raise Exception("The batch size should not be larger than the size of the dataset for testing and training set")

train_loader = Loader(
    train_dataset,
    batch_size=exp_configs["train_batch_size"],
    shuffle=True
)

logger.get().info("Loading the testing data")
test_loader = Loader(
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
    next(iter(train_loader))[0].num_node_features if MultiGpu.is_enabled() else next(iter(train_loader)).num_node_features,
    len(next(iter(train_loader))[0].y) if MultiGpu.is_enabled() else int(len(next(iter(train_loader)).y) / exp_configs["train_batch_size"]),
    next(iter(train_loader))[0].num_edge_features if MultiGpu.is_enabled() else next(iter(train_loader)).num_edge_features
)

if MultiGpu.is_enabled():
  logger.get().info("Using " + str(torch.cuda.device_count()) + " GPUs")
  exp_configs["gnn"] = DataParallelWrapper(exp_configs["gnn"])

exp_configs["gnn"] = exp_configs["gnn"].to(other_configs["device"])


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
    exp_configs["gnn"],
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

exp_configs["gnn"].eval()
with torch.no_grad():
    final_test_loss, final_accuracy, final_confusion_matrix = exp_configs["evaluator"].eval(exp_configs["gnn"], do_print=True)


#################################################
#
# VISUALISE
#
#################################################

logger.skip_line()
logger.get().info("VISUALISE")

# Ask the user to save or not the results
save_result = not other_configs["ask_for_saving"] or UserInputQuerier.ask("Save the results?")

graph_filename = other_configs["visualiser"].visualise(
    train_loss,
    test_loss,
    accuracy,
    other_configs["plot_directory_name"],
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
    experiment_results = {
        "time_taken": final_time,
        "test_loss": final_test_loss.item(),
        "train_loss": train_loss[-1].item(),
        "accuracy": final_accuracy,
        "confusion_matrix": final_confusion_matrix,
        "graph_filename": graph_filename
    }
    other_configs["save_handler"](exp_configs, experiment_results, other_configs["save_filename"]).save()


logger.skip_line()
logger.get().info("EXPERIMENT COMPLETED\n")
