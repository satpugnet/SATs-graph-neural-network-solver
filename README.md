# SATs-graph-neural-network-solver
Usage of graph neural networks to solve SATisfiability problems.

# Setup Command

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

# Generate Data

```bash
python3 src/generate_data.py
```

# Folder organisation

**out/**: Contain the outputted data.  
**scripts/**: Contains the basic scripts.  
**src/**: Contains the code.  
**src/experiments/**: Contains the file for messy experimentation of the GNN.  
**src/PyMiniSolvers/**: Contains an old fashion SATs solver.  

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