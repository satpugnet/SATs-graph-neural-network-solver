# SATs-graph-neural-network-solver
Usage of graph neural networks to solve SATisfiability problems.

# Setup and Run Command

## Run locally 

Run the setup script and install the dependencies:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
pip install -r requirements.txt
```

Then to run the application, move to the src folder and run the `main.py`:
```bash
cd src
python main.py
```

# Run on Docker

To run the code from this project on a docker container, run:
```bash
docker pull saturnin13/sats-graph-neural-network-solver
docker run -v [absolute path to experiments_results folder]:/SATs-graph-neural-network-solver/experiments_results -i -t saturnin13/sats-graph-neural-network-solver:latest /bin/bash
(i.e. docker run -v /Users/saturninpugnet/IdeaProjects/SATs-graph-neural-network-solver/experiments_results:/SATs-graph-neural-network-solver/experiments_results -i -t saturnin13/sats-graph-neural-network-solver:latest /bin/bash)
```
This will output the experiment result to the folder experiments_results on your local machine.

Then go into the src folder in the container and run the main.py. 
This can be done as follow:
```bash
cd src
python main.py
```

# Folder organisation

**data_generated/**: Contain the data generated when running the experiments.   
**scripts/**: Contains the basic scripts.  
**src/**: Contains the code.  
**src/experiments/**: Contains the list of experiment result that were saved.  
**src/PyMiniSolvers/**: Contains an old fashion SATs solver used to generate the data.  

# Instructions

## Intro

The codebase is divided into 7 parts each part of a pipeline for making an experiment with graph neural networks and SATs
problem. Each part are in a folder contains a base class that can be extended to create new modules that can be used with
the rest of the pipeline.

## Config and Main

The code is designed to be modulable and easily modifiable for the focus to be on experimenting with SATs problem and
graph neural networks.

In order to run a new experiment, you simply need to pick the configurations that you want in the `config.py` file and
run the `main.py`. The `config.py` contains instructions on how to set its parameters. It contains a section for 
each part of the pipeline.

## Modules and Pipeline

Each part is in a specific folder and has a base class that can be extended to create new class that can be used in the 
`config.py` file.
The different parts are the following:

1/ **Data generator** (in `A_data_generator`): Responsible for generating the SATs data in Dimac format and to put it in 
the `data_generated` folder.  
2/ **SAT to graph converter** (in `B_SAT_to_graph_converter`): Converts the generated SATs to a graph which can then be inputted
to a graph neural network.  
3/ **GNN** (in `C_GNN`): Creates the graph neural network architecture that will be used for the experiment.  
4/ **trainer** (in `D_trainer`): Contains the approach to train the graph neural networks.  
5/ **evaluator** (in `E_evaluator`): The way that we evaluate the results.  
6/ **visualiser** (in `F_visualiser`): How we visualise the results, currently we display the graph with the loss and accuracy.  
7/ **save** (in `G_save`): Finally, where and in what format we save the data. Currently, we save it in a file `experiments.csv`
and we save the graph.  

# Contribution

Please contribute to the code by sending pull requests, any help is appreciated, thanks in advance.

The current state of the art algorithms in the field should also be implemented in this package so that everyone get access
to them such as NeuroSAT...

# Citation

Please cite the repo with the following if it is used:
@misc{SaturninPugnet2020,  
  author = {Saturnin Pugnet},  
  title = {SATs graph neural network solver},  
  year = {2020},  
  publisher = {GitHub},  
  journal = {GitHub repository},  
  howpublished = {\url{https://github.com/saturnin13/SATs-graph-neural-network-solver}},  
  commit = {last}  
}

Copyright 2020 © Saturnin Pugnet (saturnin.13@hotmail.fr or spugnet@caltech.edu)