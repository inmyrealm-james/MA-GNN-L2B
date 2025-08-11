# GRAPH CONVOLUTIONAL NETWORK WITH MULTIPLE AGGREGATORS FOR LEARNING TO BRANCH

This repository contains the code and resources for the paper: "GRAPH CONVOLUTIONAL NETWORK WITH MULTIPLE AGGREGATORS FOR LEARNING TO BRANCH" by Nguyen Vinh Toan and Nguyen Van-Hop.

**Authors:**
*   Nguyen Vinh Toan ([nvtoan.iac@gmail.com](mailto:nvtoan.iac@gmail.com))
*   Nguyen Van-Hop ([nvhop@hcmiu.edu.vn](mailto:nvhop@hcmiu.edu.vn))

School of Industrial Engineering and Management,
International University, VNU-HCM,
Quarter 6, Linh Trung Ward, Thu Duc, HoChiMinh City, Vietnam.

## Abstract

This study introduces a Multi-Aggregator Graph Convolutional Network (MAGCN) that enhances variable selection in Mixed-Integer Linear Programming (MILP) solvers by addressing the trade-off between selection accuracy and computational efficiency inherent in traditional branching methods. Unlike prior models that rely on a single aggregation function, the proposed approach uses Principal Neighborhood Aggregation (PNA) to integrate multiple aggregators - mean, sum, max, and standard deviation - along with degree-scalers that adjust neighbor influence dynamically. A fusion Multi-Layer Perceptron (MLP) and skip-connection technique further improve feature propagation and stabilize learning. The model, trained through imitation learning based on strong branching decisions and integrated within a Branch-and-Bound framework, effectively minimizes the number of explored nodes and LP relaxations. Benchmark tests on various MILP instances, including variants of the Multi-Traveling Salesman Problem and Job Shop Scheduling Problem, reveal competitive or superior performance compared to conventional and attention-enhanced GCN architectures in solving complex MILP problems.


## Setup and Installation

This project uses Conda for environment management and several specialized libraries. The `02.main.ipynb` notebook handles the setup within a Google Colab environment.

**Key Dependencies:**

*   Python 3.x
*   NumPy (`==1.26.4` specifically, due to potential compatibility issues otherwise addressed in the notebook)
*   PyTorch (`torch`, `torchvision`, `torchaudio`)
*   PyTorch Geometric (`torch-geometric`)
*   Ecole (`ecole==0.8.2` or as installed by Conda)
*   SCIP (`scip=8.0` or as installed by Conda)
*   PySCIPOpt (`pyscipopt`)
*   Psutil
*   Condacolab (for Colab Conda environment)
*   Mamba (for faster Conda installations)
*   Scipy

**To replicate the environment (especially outside Colab):**

1.  **Install Conda/Miniconda.**
2.  It's recommended to create a Conda environment:
    ```bash
    conda create -n 02.main python=3.9  # Or your preferred Python 3 version
    conda activate 02.main
    ```
3.  Install Mamba for faster package installation (optional but recommended):
    ```bash
    conda install -c conda-forge mamba
    ```
4.  Install core dependencies using Conda/Mamba (refer to the notebook for exact commands):
    ```bash
    mamba install -c conda-forge ecole scip=8.0 pyscipopt
    ```
5.  Install PyTorch (select the version appropriate for your CUDA setup if using GPU):
    ```bash
    # Example for CUDA 11.8 - check PyTorch website for current commands
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
6.  Install PyTorch Geometric and other pip packages:
    ```bash
    pip install torch-geometric numpy==1.26.4 psutil scipy
    ```
7.  If running in Google Colab, the "SET UP" section of `02.main.ipynb` will handle this using `condacolab`.

**Note on NumPy version:** The notebook explicitly uninstalls the default NumPy and installs `numpy==1.26.4`. This might be crucial for compatibility.

## Usage

The `02.main.ipynb` notebook demonstrates the full pipeline. Below are conceptual steps, which are implemented within the notebook or could be run as standalone scripts.

### 1. Generate MILP Instances 

If you need to generate your own MILP problem instances (`.lp` files):
*   Use the `01.generate_instances.ipynb` notebook.
*   This notebook contains functions like `generate_Standard_MTSP`, `generate_MinMax_MTSP`, `generate_Bounded_MTSP`, `generate_JSSP`, etc.
*   Specify the problem type, size, and output directory (e.g., `data/instances/`).
*   Example structure from the notebook:
    ```python
    # In Generate_Instances.ipynb
    # args = parser.parse_args(['Standard_MTSP']) # or 'JSSP', etc.
    # ... logic to generate instances ...
    # generate_Standard_MTSP(rng, filename, n_customers=ncs, m_salesman=nsm)
    ```

### 2. Generate Training Data (Samples)

This step uses Ecole to collect (state, action) pairs from solving MILP instances with an expert brancher (Strong Branching).
*   This is handled in the "Generate Dataset" section of `02.main.ipynb`.
*   It expects `.lp` instance files in `data/instances/PROBLEM_TYPE/...`.
*   It outputs `.pkl` sample files to `data/samples/PROBLEM_TYPE/...`.
*   Key parameters: problem type, number of samples, number of jobs, time limit.
    ```python
    # In 02.main.ipynb, under "Generate Dataset"
    # args = parser.parse_args(['JSSP']) # or other problem types
    # ...
    # collect_samples(instances_train, out_dir + '/train/1', rng, train_size, ...)
    ```

### 3. Train the GNN Model

This involves pre-training normalization layers and then training the GNN policy.
*   This is handled in the "Train GNN" section of `02.main.ipynb` (with "Pre-Train" and "Training Loop" subsections).
*   It loads `.pkl` samples from `data/samples/`.
*   It saves model checkpoints (`.pkl`) and logs (`.txt`) to `model/PROBLEM_TYPE/SEED/`.
*   Key parameters: problem type, seed, GPU ID, learning rate, batch size.
    ```python
    # In 02.main.ipynb, under "Train GNN" -> "Pre-Train" and "Training Loop"
    # args = parser.parse_args(['Standard_MTSP']) # or other problem types
    # ...
    # policy = GNNPolicy(avg_d_left=avg_log_left, avg_d_right=avg_log_right).to(device)

    # ...
    # # Pre-training
    # n = pretrain(policy, pretrain_loader)
    # # Training loop
    # train_loss, train_kacc, entropy = process(policy, train_loader, top_k, optimizer)
    ```

### 4. Evaluate the Trained Model

Compare the performance of the trained GNN model(s) against baselines on test instances.
*   This is handled in the "Evaluate" section of `02.main.ipynb`.
*   It loads trained model checkpoints from `model/PROBLEM_TYPE/SEED/`.
*   It solves test instances from `data/instances/` using the GNN policy.
*   It outputs results (nodes, LPs, time, gap) to a CSV file in the `results/` directory.
*   Key parameters: problem type, GPU ID, time limit.
    ```python
    # In 02.main.ipynb, under "Evaluate"
    # args = parser.parse_args(['Standard_MTSP']) # or other problem types
    # ...
    # model_path = "/content/drive/MyDrive/Thesis/model/JSSP/0/JSSP_PNA_train_PARAM.pkl" # Example
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # ...
    # # Evaluation loop
    # observation, action_set, _, done, _ = env.reset(instance['path'])
    # while not done:
    #     logits = policy['model'](*observation_tensor)
    #     action = action_set[logits[action_set.astype(np.int64)].argmax()]
    #     observation, action_set, _, done, _ = env.step(action)
    ```

## Models Implemented

The repository includes implementations for:

**MAGCN (Multi-Aggregator Graph Convolutional Network):** The proposed model from our paper. It uses Principal Neighborhood Aggregation (PNA) with multiple aggregators (mean, sum, max, std) and degree scalers, along with a fusion MLP and skip-connections.
    *   Class: `GNNPolicy` in the MAGCN model section of the notebook, utilizing `MultiAggregatorBipartiteConvManual_PNA`.
